import operator
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.sparse as sps
from implicit.cpu.als import AlternatingLeastSquares

from recsys_tbank.split import DAY, THRESHOLD, Frame, assert_no_future, novel_flag

SOURCES = ("user", "pop", "als", "i2i")
SLICES = ("all", "novel", "repeat")
LABELS = ("y_org", "y_all")
I2I_BATCH = 64
ALS_DOT_BATCH = 1 << 18

_PAIR = ["uid", "item_id"]
_SCORED = ("als", "i2i")


@dataclass(frozen=True)
class CandidateConfig:
    user_top_n: int = 200
    user_days: int = 90
    pop_top_n: int = 100
    pop_days: int = 7
    als_top_n: int = 100
    als_factors: int = 64
    als_iterations: int = 15
    als_regularization: float = 0.01
    als_alpha: float = 1.0
    i2i_anchor_items: int = 50
    i2i_top_n: int = 100
    seed: int = 42

    def config_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = CandidateConfig()


class ALSFactors(NamedTuple):
    uids: np.ndarray
    items: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray


def _source_schema(name: str) -> dict:
    schema = {"uid": pl.UInt32, "item_id": pl.UInt32, f"rank_{name}": pl.UInt32}
    if name in _SCORED:
        schema[f"score_{name}"] = pl.Float32
    return schema


def _empty(name: str) -> pl.DataFrame:
    return pl.DataFrame(schema=_source_schema(name))


def _finish(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    return frame.select(
        pl.col(c).cast(t) for c, t in _source_schema(name).items()
    ).sort(_PAIR)


def _rank_within_user(
    frame: pl.DataFrame, by: list[str], descending: list[bool], n: int, name: str
) -> pl.DataFrame:
    return (
        frame.sort(["uid", *by], descending=[False, *descending])
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.UInt32)
            .over("uid")
            .alias(f"rank_{name}")
        )
        .filter(pl.col(f"rank_{name}") <= n)
    )


def _user_source(listens: pl.DataFrame, T: int, cfg: CandidateConfig) -> pl.DataFrame:
    counts = (
        listens.filter(pl.col("ts") >= T - cfg.user_days * DAY)
        .group_by(_PAIR)
        .agg(pl.len().alias("n"), pl.col("ts").max().alias("last"))
    )
    ranked = _rank_within_user(
        counts, ["n", "last", "item_id"], [True, True, False], cfg.user_top_n, "user"
    )
    return _finish(ranked, "user")


def _pop_source(
    listens: pl.DataFrame, users: pl.DataFrame, T: int, cfg: CandidateConfig
) -> pl.DataFrame:
    top = (
        listens.filter(pl.col("ts") >= T - cfg.pop_days * DAY)
        .group_by("item_id")
        .agg(pl.len().alias("n"))
        .sort(["n", "item_id"], descending=[True, False])
        .head(cfg.pop_top_n)
        .with_row_index("rank_pop", offset=1)
        .select("item_id", "rank_pop")
    )
    return _finish(users.join(top, how="cross"), "pop")


def _listens(lazy: pl.LazyFrame) -> pl.DataFrame:
    return (
        lazy.filter(pl.col("event_type") == "listen")
        .select(
            pl.col("uid").cast(pl.UInt32),
            pl.col("item_id").cast(pl.UInt32),
            pl.col("timestamp").cast(pl.Int64).alias("ts"),
            "played_ratio_pct",
        )
        .collect()
    )


def _index(values: pl.Series, name: str) -> pl.DataFrame:
    return pl.DataFrame({name: values}).with_row_index(f"{name}_idx")


def _als_matrix(
    listens: pl.DataFrame, cfg: CandidateConfig
) -> tuple[np.ndarray, np.ndarray, sps.csr_matrix]:
    pos = (
        listens.filter(pl.col("played_ratio_pct") >= THRESHOLD)
        .group_by(_PAIR)
        .agg(pl.len().alias("n"))
        .sort(_PAIR)
    )
    uids = pos["uid"].unique().sort()
    items = pos["item_id"].unique().sort()
    pos = pos.join(_index(uids, "uid"), on="uid").join(
        _index(items, "item_id"), on="item_id"
    )
    conf = (
        1.0 + cfg.als_alpha * np.log1p(pos["n"].to_numpy().astype(np.float64))
    ).astype(np.float32)
    matrix = sps.csr_matrix(
        (conf, (pos["uid_idx"].to_numpy(), pos["item_id_idx"].to_numpy())),
        shape=(uids.len(), items.len()),
        dtype=np.float32,
    )
    matrix.sort_indices()
    return uids.to_numpy(), items.to_numpy(), matrix


def _fit_als(
    listens: pl.DataFrame, cfg: CandidateConfig
) -> tuple[np.ndarray, np.ndarray, AlternatingLeastSquares | None]:
    uids, items, matrix = _als_matrix(listens, cfg)
    if matrix.nnz == 0:
        return uids, items, None
    model = AlternatingLeastSquares(
        factors=cfg.als_factors,
        regularization=cfg.als_regularization,
        alpha=1.0,
        dtype=np.float32,
        iterations=cfg.als_iterations,
        calculate_training_loss=False,
        num_threads=1,
        random_state=cfg.seed,
    )
    model.fit(matrix, show_progress=False)
    return uids, items, model


def als_factors(
    hist: Frame, T: int, cfg: CandidateConfig = DEFAULT_CONFIG
) -> ALSFactors:
    assert_no_future(hist, T)
    uids, items, model = _fit_als(_listens(hist.lazy()), cfg)
    if model is None:
        empty = np.zeros((0, cfg.als_factors), dtype=np.float32)
        return ALSFactors(uids, items, empty, empty)
    return ALSFactors(uids, items, model.user_factors, model.item_factors)


def _lookup(index: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if index.size == 0:
        return np.zeros(values.size, dtype=np.int64), np.zeros(values.size, dtype=bool)
    pos = np.minimum(np.searchsorted(index, values), index.size - 1)
    return pos, index[pos] == values


def als_dot(factors: ALSFactors, pairs: pl.DataFrame) -> pl.Series:
    u, u_ok = _lookup(factors.uids, pairs["uid"].cast(pl.UInt32).to_numpy())
    i, i_ok = _lookup(factors.items, pairs["item_id"].cast(pl.UInt32).to_numpy())
    ok = u_ok & i_ok
    rows = np.flatnonzero(ok)
    scores = np.zeros(ok.size, dtype=np.float32)
    for start in range(0, rows.size, ALS_DOT_BATCH):
        r = rows[start : start + ALS_DOT_BATCH]
        scores[r] = np.einsum(
            "ij,ij->i", factors.user_factors[u[r]], factors.item_factors[i[r]]
        )
    return (
        pl.DataFrame({"score": scores, "ok": ok})
        .select(pl.when("ok").then("score").alias("als_dot"))
        .to_series()
    )


def _als_source(listens: pl.DataFrame, cfg: CandidateConfig) -> pl.DataFrame:
    uids, items, model = _fit_als(listens, cfg)
    if model is None:
        return _empty("als")
    n = min(cfg.als_top_n, items.size)
    ids, scores = model.recommend(
        np.arange(uids.size), None, N=n, filter_already_liked_items=False
    )
    ids = np.asarray(ids).reshape(uids.size, -1)
    scores = np.asarray(scores, dtype=np.float32).reshape(uids.size, -1)
    keep = (ids >= 0) & np.isfinite(scores)
    rows = np.broadcast_to(np.arange(uids.size)[:, None], ids.shape)[keep]
    frame = pl.DataFrame(
        {
            "uid": uids[rows],
            "item_id": items[ids[keep]],
            "score_als": scores[keep],
        }
    )
    ranked = _rank_within_user(frame, ["score_als", "item_id"], [True, False], n, "als")
    return _finish(ranked, "als")


def _i2i_source(listens: pl.DataFrame, cfg: CandidateConfig) -> pl.DataFrame:
    if listens.height == 0:
        return _empty("i2i")
    items = listens["item_id"].unique().sort()
    item_idx = _index(items, "item_id")
    pos = (
        listens.filter(pl.col("played_ratio_pct") >= THRESHOLD)
        .select(_PAIR)
        .unique()
        .sort(_PAIR)
    )
    pos_users = pos["uid"].unique().sort()
    pos = pos.join(_index(pos_users, "uid"), on="uid").join(item_idx, on="item_id")
    binary = sps.csr_matrix(
        (
            np.ones(pos.height, dtype=np.float64),
            (pos["uid_idx"].to_numpy(), pos["item_id_idx"].to_numpy()),
        ),
        shape=(pos_users.len(), items.len()),
    )
    norms = np.sqrt(np.asarray(binary.sum(axis=0)).ravel())
    inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    item_user = (binary @ sps.diags(inv)).T.tocsr()
    item_user.sort_indices()
    anchors = _rank_within_user(
        listens.group_by(_PAIR).agg(pl.col("ts").max().alias("last")),
        ["last", "item_id"],
        [True, False],
        cfg.i2i_anchor_items,
        "anchor",
    ).join(item_idx, on="item_id")
    anchor_users = anchors["uid"].unique().sort()
    anchors = anchors.join(_index(anchor_users, "uid"), on="uid").sort(_PAIR)
    anchor_matrix = sps.csr_matrix(
        (
            np.ones(anchors.height, dtype=np.float64),
            (anchors["uid_idx"].to_numpy(), anchors["item_id_idx"].to_numpy()),
        ),
        shape=(anchor_users.len(), items.len()),
    )
    anchor_matrix.sort_indices()
    item_ids = items.to_numpy()
    out_uid, out_item, out_score = [], [], []
    for start in range(0, anchor_users.len(), I2I_BATCH):
        block = anchor_matrix[start : start + I2I_BATCH]
        via_users = (block @ item_user).toarray()
        scores = np.ascontiguousarray(np.asarray(item_user @ via_users.T).T)
        for j in range(block.shape[0]):
            col = scores[j]
            col[block.indices[block.indptr[j] : block.indptr[j + 1]]] = 0.0
            cand = np.flatnonzero(col > 0)
            if cand.size > cfg.i2i_top_n:
                vals = col[cand]
                kth = np.partition(vals, vals.size - cfg.i2i_top_n)[
                    vals.size - cfg.i2i_top_n
                ]
                cand = cand[vals >= kth]
                order = np.lexsort((cand, -col[cand]))
                cand = cand[order[: cfg.i2i_top_n]]
            out_uid.append(np.full(cand.size, anchor_users[start + j], dtype=np.uint32))
            out_item.append(item_ids[cand])
            out_score.append(col[cand])
    if not out_uid:
        return _empty("i2i")
    frame = pl.DataFrame(
        {
            "uid": np.concatenate(out_uid),
            "item_id": np.concatenate(out_item),
            "_score": np.concatenate(out_score),
        }
    )
    ranked = _rank_within_user(
        frame, ["_score", "item_id"], [True, False], cfg.i2i_top_n, "i2i"
    ).with_columns(pl.col("_score").cast(pl.Float32).alias("score_i2i"))
    return _finish(ranked, "i2i")


def build_candidates(
    hist: Frame,
    T: int,
    cfg: CandidateConfig = DEFAULT_CONFIG,
    sources: Sequence[str] = SOURCES,
) -> pl.DataFrame:
    assert_no_future(hist, T)
    T = operator.index(T)
    unknown = set(sources) - set(SOURCES)
    if unknown or not sources:
        raise ValueError(f"sources must be a non-empty subset of {SOURCES}")
    lazy = hist.lazy()
    users = lazy.select(pl.col("uid").cast(pl.UInt32)).unique().sort("uid").collect()
    listens = _listens(lazy)
    builders = {
        "user": lambda: _user_source(listens, T, cfg),
        "pop": lambda: _pop_source(listens, users, T, cfg),
        "als": lambda: _als_source(listens, cfg),
        "i2i": lambda: _i2i_source(listens, cfg),
    }
    parts = {s: builders[s]() if s in sources else _empty(s) for s in SOURCES}
    out = pl.concat([p.select(_PAIR) for p in parts.values()]).unique().sort(_PAIR)
    for part in parts.values():
        out = out.join(part, on=_PAIR, how="left", maintain_order="left")
    flags = [
        pl.col(f"rank_{s}").is_not_null().cast(pl.UInt8).alias(f"in_{s}")
        for s in SOURCES
    ]
    out = out.with_columns(flags)
    out = out.with_columns(
        pl.sum_horizontal(f"in_{s}" for s in SOURCES).cast(pl.UInt8).alias("n_sources")
    )
    return out.select(
        *_PAIR,
        *(f"in_{s}" for s in SOURCES),
        *(f"rank_{s}" for s in SOURCES),
        *(f"score_{s}" for s in _SCORED),
        "n_sources",
    )


def attach_labels(
    cands: pl.DataFrame, hist: Frame, pairs: pl.DataFrame
) -> pl.DataFrame:
    labels = pairs.select(
        pl.col("uid").cast(pl.UInt32),
        pl.col("item_id").cast(pl.UInt32),
        *(pl.col(c).cast(pl.UInt8) for c in LABELS),
    )
    labeled = cands.join(labels, on=_PAIR, how="left", maintain_order="left")
    labeled = labeled.with_columns(pl.col(c).fill_null(0) for c in LABELS)
    return novel_flag(hist, labeled).with_columns(pl.col("novel").cast(pl.UInt8))


def _check_label(label: str) -> None:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")


def n_pos_table(pairs_with_novel: pl.DataFrame, label: str) -> pl.DataFrame:
    _check_label(label)
    return (
        pairs_with_novel.filter(pl.col(label) == 1)
        .group_by(pl.col("uid").cast(pl.UInt32))
        .agg(
            pl.len().cast(pl.Int64).alias("all"),
            pl.col("novel").cast(pl.Int64).sum().alias("novel"),
        )
        .with_columns((pl.col("all") - pl.col("novel")).alias("repeat"))
        .select("uid", *SLICES)
        .sort("uid")
    )


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def source_report(
    cands_with_labels: pl.DataFrame, n_pos: pl.DataFrame, label: str
) -> pl.DataFrame:
    _check_label(label)
    slice_filters = {
        "all": None,
        "novel": pl.col("novel") == 1,
        "repeat": pl.col("novel") == 0,
    }
    rows = []
    for slice_name, row_filter in slice_filters.items():
        sub = (
            cands_with_labels
            if row_filter is None
            else cands_with_labels.filter(row_filter)
        )
        y = sub[label].to_numpy().astype(bool)
        total = int(n_pos[slice_name].sum())
        cum = np.zeros(sub.height, dtype=bool)
        prev_recall = 0.0
        masks = [(s, sub[f"in_{s}"].to_numpy().astype(bool)) for s in SOURCES]
        masks.append(("union", np.ones(sub.height, dtype=bool)))
        for source, mask in masks:
            found = int((y & mask).sum())
            recall = _ratio(found, total)
            row = {
                "source": source,
                "slice": slice_name,
                "rows": int(mask.sum()),
                "pos_found": found,
                "pos_total": total,
                "recall": recall,
                "pos_rate": _ratio(found, int(mask.sum())),
                "cum_recall": recall,
                "inc_recall": None,
            }
            if source != "union":
                cum |= mask
                row["cum_recall"] = _ratio(int((y & cum).sum()), total)
                row["inc_recall"] = row["cum_recall"] - prev_recall
                prev_recall = row["cum_recall"]
            rows.append(row)
    schema = {
        "source": pl.String,
        "slice": pl.String,
        "rows": pl.Int64,
        "pos_found": pl.Int64,
        "pos_total": pl.Int64,
        "recall": pl.Float64,
        "pos_rate": pl.Float64,
        "cum_recall": pl.Float64,
        "inc_recall": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)
