from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl


def _check_frames(frame: pl.DataFrame, n_pos: pl.DataFrame) -> pl.DataFrame:
    if frame.select(pl.struct("uid", "item_id").is_duplicated().any()).item():
        raise ValueError("duplicate (uid, item_id) rows")
    if n_pos["uid"].is_duplicated().any():
        raise ValueError("duplicate uid in n_pos")
    if frame.height and not frame["label"].is_in([0, 1]).all():
        raise ValueError("label must be 0/1")
    n_pos = n_pos.select(
        pl.col("uid").cast(frame.schema["uid"]), pl.col("n_pos").cast(pl.Int64)
    )
    found = frame.group_by("uid").agg(
        pl.col("label").cast(pl.Int64).sum().alias("found")
    )
    bad = n_pos.join(found, on="uid", how="inner").filter(
        pl.col("found") > pl.col("n_pos")
    )
    if bad.height:
        raise ValueError("more positive candidates than n_pos for some users")
    return n_pos


def rank_metrics(
    scored: pl.DataFrame, n_pos: pl.DataFrame, ks: Sequence[int] = (10, 100)
) -> dict[str, float]:
    if (
        scored.height
        and scored.select(
            pl.col("score").is_null().any() | pl.col("score").is_nan().any()
        ).item()
    ):
        raise ValueError("score has null or NaN")
    n_pos = _check_frames(scored, n_pos)
    users = n_pos.filter(pl.col("n_pos") >= 1).sort("uid")
    ranked = (
        scored.join(users.select("uid"), on="uid", how="semi")
        .sort(["uid", "score", "item_id"], descending=[False, True, False])
        .with_columns(rank=pl.int_range(1, pl.len() + 1).over("uid"))
    )
    gain = pl.col("label").cast(pl.Float64)
    aggs = []
    for k in ks:
        top = pl.col("rank") <= k
        aggs.append((gain * top).sum().alias(f"hits_{k}"))
        aggs.append((gain * top / (pl.col("rank") + 1).log(2)).sum().alias(f"dcg_{k}"))
    per_user = (
        users.join(ranked.group_by("uid").agg(aggs), on="uid", how="left")
        .fill_null(0)
        .sort("uid")
    )
    npos = per_user["n_pos"].to_numpy()
    out: dict[str, float] = {}
    for k in ks:
        hits = per_user[f"hits_{k}"].to_numpy()
        dcg = per_user[f"dcg_{k}"].to_numpy()
        ideal = np.concatenate(([0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))))
        capped = np.minimum(npos, k)
        out[f"ndcg@{k}"] = _mean(dcg / np.where(capped > 0, ideal[capped], 1.0))
        out[f"recall@{k}"] = _mean(hits / np.maximum(npos, 1))
        out[f"recall_cap@{k}"] = _mean(hits / np.maximum(capped, 1))
        out[f"hit_rate@{k}"] = _mean((hits > 0).astype(np.float64))
    out["n_users"] = users.height
    return out


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def candidate_recall(candidates: pl.DataFrame, n_pos: pl.DataFrame) -> dict[str, float]:
    n_pos = _check_frames(candidates, n_pos)
    users = n_pos.filter(pl.col("n_pos") >= 1).sort("uid")
    found = candidates.group_by("uid").agg(
        pl.col("label").cast(pl.Int64).sum().alias("found")
    )
    per_user = users.join(found, on="uid", how="left").fill_null(0).sort("uid")
    hits = per_user["found"].to_numpy()
    npos = per_user["n_pos"].to_numpy()
    total = int(npos.sum())
    return {
        "n_rows": candidates.height,
        "n_pos_found": int(hits.sum()),
        "n_pos_total": total,
        "recall_micro": hits.sum() / total if total else float("nan"),
        "recall_macro": _mean(hits / np.maximum(npos, 1)),
        "n_users": users.height,
    }


def logloss(y, p, eps: float = 1e-15) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def roc_auc(y, score) -> float:
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    n1 = y.sum()
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_score[1:] != sorted_score[:-1]))
    )
    ends = np.concatenate((starts[1:], [score.size]))
    avg_rank = (starts + ends + 1) / 2.0
    ranks = np.empty(score.size)
    ranks[order] = np.repeat(avg_rank, ends - starts)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@dataclass(frozen=True)
class BinCalibrator:
    edges: np.ndarray
    rates: np.ndarray

    def transform(self, score) -> np.ndarray:
        idx = np.searchsorted(
            self.edges, np.asarray(score, dtype=np.float64), side="right"
        )
        return self.rates[idx]


def _pool_adjacent_violators(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks: list[list[float]] = []
    for v, w in zip(values, weights):
        blocks.append([v * w, w, 1])
        while (
            len(blocks) > 1
            and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]
        ):
            s, w2, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += w2
            blocks[-1][2] += c
    return np.concatenate([np.full(c, s / w) for s, w, c in blocks])


def bin_calibrator(
    score_train, y_train, bins: int = 20, alpha: float = 1.0
) -> BinCalibrator:
    score = np.asarray(score_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if score.size == 0 or score.size != y.size or np.isnan(score).any():
        raise ValueError(
            "score_train and y_train must be non-empty, equal length, without NaN"
        )
    edges = np.unique(np.quantile(score, np.linspace(0.0, 1.0, bins + 1))[1:-1])
    occupied = np.unique(np.searchsorted(edges, score, side="right"))
    edges = edges[occupied[1:] - 1]
    idx = np.searchsorted(edges, score, side="right")
    n = np.bincount(idx, minlength=edges.size + 1).astype(np.float64)
    pos = np.bincount(idx, weights=y, minlength=edges.size + 1)
    prior = (y.sum() + 1) / (y.size + 2)
    rates = (pos + alpha * prior) / (n + alpha)
    return BinCalibrator(edges=edges, rates=_pool_adjacent_violators(rates, n))


def evaluate(
    scored: pl.DataFrame,
    n_pos: pl.DataFrame,
    slices: Mapping[str, pl.Expr | None],
    ks: Sequence[int] = (10, 100),
    prob_col: str | None = None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, row_filter in slices.items():
        sub = scored if row_filter is None else scored.filter(row_filter)
        slice_pos = n_pos.select("uid", pl.col(name).alias("n_pos"))
        res = rank_metrics(sub, slice_pos, ks)
        cand = candidate_recall(sub, slice_pos)
        res["n_rows"] = cand["n_rows"]
        res["cand_recall_micro"] = cand["recall_micro"]
        res["cand_recall_macro"] = cand["recall_macro"]
        y = sub["label"].to_numpy()
        res["roc_auc"] = roc_auc(y, sub["score"].to_numpy())
        if prob_col is not None:
            res["logloss"] = logloss(y, sub[prob_col].to_numpy())
        out[name] = res
    return out
