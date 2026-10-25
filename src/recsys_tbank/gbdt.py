import hashlib
import json
import re
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import lightgbm as lgb
import numpy as np
import polars as pl

from recsys_tbank import baselines as bl
from recsys_tbank import features as ft
from recsys_tbank.candidates import LABELS, SLICES, SOURCES, attach_labels
from recsys_tbank.dataset import (
    CAND_PREFIX,
    GROUP_KEYS,
    RESERVED,
    WindowData,
    group_n_pos,
    group_sizes,
    stack_frames,
)
from recsys_tbank.metrics import bin_calibrator
from recsys_tbank.split import Frame, get_window, history

SEED = 42
GIT_COMMIT = bl.GIT_COMMIT
OBJECTIVES = ("lambdarank", "binary")
PROTOCOLS = ("inner", "outer")
SELECTION_WINDOW = "train_280"
FORBIDDEN_SELECTION = ("val", "test")
INNER_TRAIN_WINDOWS = ("train_266", "train_273")
OUTER_TARGET = "val"
REFIT_FACTOR = 1.5
STOP_K = 10
STOP_METRIC = f"ndcg@{STOP_K}"
CALIB_BINS = bl.CALIB_BINS
CALIB_FOLDS = 2
DEFAULT_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
}
RESERVED_PARAMS = (
    "objective",
    "metric",
    "seed",
    "random_seed",
    "random_state",
    "num_threads",
    "n_jobs",
    "deterministic",
    "force_col_wise",
    "verbosity",
    "lambdarank_truncation_level",
    "num_iterations",
    "num_boost_round",
    "n_estimators",
    "early_stopping_round",
    "early_stopping_rounds",
)
VOLATILE_KEYS = (*bl.VOLATILE_KEYS, "resources")
BASELINE_PARAMS = bl.BaselineParams(
    b1_variant="all", b2_variant="all", b2_days=30, b3_variant="org"
)
BASELINE_PARAMS_SOURCE = (
    "runs/20261018-210235-baselines.json config.baseline_params "
    "(Phase 1 choice under its choice_rule, frozen)"
)
SCORED_BASELINES = ("B0", "B1", "B2", "B3", "B5", "B6")


def check_selection_window(name: str) -> str:
    if name in FORBIDDEN_SELECTION:
        raise ValueError(f"selection_window must never be {name!r}")
    get_window(name)
    return name


@dataclass(frozen=True)
class GBDTConfig:
    objective: str = "lambdarank"
    label: str = "y_org"
    params: Mapping[str, object] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    seed: int = SEED
    num_threads: int = 8
    deterministic: bool = True
    force_col_wise: bool = True
    groups: tuple[str, ...] | None = None
    exclude_groups: tuple[str, ...] = ()
    features: tuple[str, ...] | None = None
    lambdarank_truncation_level: int = 30
    train_windows: tuple[str, ...] | None = None
    selection_window: str = SELECTION_WINDOW

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        if self.groups is not None and self.features is not None:
            raise ValueError("give either groups or features, not both")
        clash = sorted(set(self.params) & set(RESERVED_PARAMS))
        if clash:
            raise ValueError(f"params must not set {clash}; use GBDTConfig fields")
        if self.num_boost_round <= 0 or self.early_stopping_rounds < 0:
            raise ValueError("num_boost_round > 0 and early_stopping_rounds >= 0")
        check_selection_window(self.selection_window)

    def config_dict(self) -> dict:
        out = asdict(self)
        out["params"] = dict(self.params)
        return out

    def lgb_params(self) -> dict:
        out = {
            **self.params,
            "objective": self.objective,
            "metric": "None",
            "seed": self.seed,
            "num_threads": self.num_threads,
            "deterministic": self.deterministic,
            "force_col_wise": self.force_col_wise,
            "verbosity": -1,
        }
        if self.objective == "lambdarank":
            out["lambdarank_truncation_level"] = self.lambdarank_truncation_level
        return out


def resolve_features(
    cfg: GBDTConfig, specs: Sequence[ft.FeatureSpec] | None = None
) -> tuple[str, ...]:
    specs = ft.all_specs() if specs is None else tuple(specs)
    registered = {s.group for s in specs}
    if cfg.features is not None:
        known = {s.name for s in specs}
        unknown = [f for f in cfg.features if f not in known]
        if unknown:
            raise ValueError(f"unknown features: {unknown}")
        wanted = set(cfg.features)
        names = tuple(s.name for s in specs if s.name in wanted)
    else:
        groups = registered if cfg.groups is None else set(cfg.groups)
        unknown = sorted((groups | set(cfg.exclude_groups)) - registered)
        if unknown:
            raise ValueError(f"unregistered feature groups: {unknown}")
        keep = groups - set(cfg.exclude_groups)
        names = tuple(s.name for s in specs if s.group in keep)
    if not names:
        raise ValueError("empty feature set")
    return names


def feature_groups(names: Sequence[str]) -> dict[str, list[str]]:
    owner = {s.name: s.group for s in ft.all_specs()}
    out: dict[str, list[str]] = {}
    for name in names:
        out.setdefault(owner.get(name, "other"), []).append(name)
    return out


class FitResult(NamedTuple):
    booster: lgb.Booster
    cfg: GBDTConfig
    features: tuple[str, ...]
    best_iteration: int
    best_score: float | None
    train_rows: int
    train_groups: int
    dropped_groups: int
    curve: tuple[float, ...]


def _check_columns(frame: pl.DataFrame, features: Sequence[str], label: str) -> None:
    missing = [c for c in (*GROUP_KEYS, "item_id", label, *features) if c not in frame]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    bad = [f for f in features if f in RESERVED or f.startswith(CAND_PREFIX)]
    if bad:
        raise ValueError(
            f"features must not be keys, labels or candidate columns: {bad}"
        )


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return frame.select(pl.col(f).cast(pl.Float32) for f in features).to_numpy()


def training_rows(frame: pl.DataFrame, cfg: GBDTConfig) -> tuple[pl.DataFrame, int]:
    if cfg.objective == "binary":
        return frame, 0
    total = group_sizes(frame).size
    pos = pl.col(cfg.label).cast(pl.Int64).sum().over(list(GROUP_KEYS))
    kept = frame.filter(pos > 0)
    return kept, total - group_sizes(kept).size


class RankEval:
    def __init__(
        self,
        frame: pl.DataFrame,
        label: str,
        n_pos: pl.DataFrame | None = None,
        k: int = STOP_K,
    ):
        sizes = group_sizes(frame)
        self.k = k
        self.n = frame.height
        self.gid = np.repeat(np.arange(sizes.size), sizes)
        self.start = (np.cumsum(sizes) - sizes).astype(np.int64)
        self.y = frame[label].cast(pl.Float64).to_numpy()
        found = np.bincount(self.gid, weights=self.y, minlength=sizes.size)
        extra = 0
        if n_pos is None:
            npos = found
        else:
            heads = frame.select(
                pl.col("T").cast(pl.Int64).gather(self.start),
                pl.col("uid").cast(pl.UInt32).gather(self.start),
            )
            table = n_pos.select(
                pl.col("T").cast(pl.Int64),
                pl.col("uid").cast(pl.UInt32),
                pl.col("n_pos").cast(pl.Int64),
            )
            if table.select(list(GROUP_KEYS)).is_duplicated().any():
                raise ValueError("duplicate (T, uid) in n_pos")
            joined = heads.join(
                table, on=list(GROUP_KEYS), how="left", maintain_order="left"
            )
            npos = joined["n_pos"].fill_null(0).to_numpy()
            if (found > npos).any():
                raise ValueError("more positive candidates than n_pos for some groups")
            extra = (
                table.filter(pl.col("n_pos") >= 1)
                .join(heads, on=list(GROUP_KEYS), how="anti")
                .height
            )
        npos = np.asarray(npos).astype(np.int64)
        ideal = np.concatenate(([0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))))
        self.ok = npos >= 1
        self.idcg = ideal[np.minimum(npos[self.ok], k)]
        self.denom = int(self.ok.sum()) + extra

    def ndcg(self, scores) -> float:
        if not self.denom:
            return float("nan")
        scores = np.asarray(scores, dtype=np.float64)
        order = np.lexsort((-scores, self.gid))
        g = self.gid[order]
        rank = np.arange(self.n) - self.start[g] + 1
        gain = np.where(rank <= self.k, self.y[order] / np.log2(rank + 1.0), 0.0)
        dcg = np.bincount(g, weights=gain, minlength=self.ok.size)
        return float((dcg[self.ok] / self.idcg).sum() / self.denom)


def _dataset(frame: pl.DataFrame, features, cfg: GBDTConfig, reference=None):
    group = group_sizes(frame) if cfg.objective == "lambdarank" else None
    return lgb.Dataset(
        _matrix(frame, features),
        label=frame[cfg.label].cast(pl.Float64).to_numpy(),
        group=group,
        feature_name=list(features),
        reference=reference,
        free_raw_data=True,
    )


def fit(
    train_frame: pl.DataFrame,
    valid_frame: pl.DataFrame | None,
    cfg: GBDTConfig,
    valid_n_pos: pl.DataFrame | None = None,
    features: Sequence[str] | None = None,
) -> FitResult:
    features = resolve_features(cfg) if features is None else tuple(features)
    _check_columns(train_frame, features, cfg.label)
    kept, dropped = training_rows(train_frame, cfg)
    if kept.height == 0:
        raise ValueError("no training rows left")
    dtrain = _dataset(kept, features, cfg)
    curve: list[float] = []
    kwargs: dict = {}
    if valid_frame is not None:
        _check_columns(valid_frame, features, cfg.label)
        ev = RankEval(valid_frame, cfg.label, valid_n_pos)

        def feval(preds, _data):
            value = ev.ndcg(preds)
            curve.append(value)
            return STOP_METRIC, value, True

        callbacks = []
        if cfg.early_stopping_rounds:
            callbacks.append(
                lgb.early_stopping(
                    cfg.early_stopping_rounds, first_metric_only=True, verbose=False
                )
            )
        kwargs = {
            "valid_sets": [_dataset(valid_frame, features, cfg, dtrain)],
            "valid_names": [cfg.selection_window],
            "feval": feval,
            "callbacks": callbacks,
        }
    booster = lgb.train(
        cfg.lgb_params(), dtrain, num_boost_round=cfg.num_boost_round, **kwargs
    )
    if curve:
        best = int(np.argmax(curve)) + 1
        best_score = curve[best - 1]
    else:
        best, best_score = booster.current_iteration(), None
    return FitResult(
        booster=booster,
        cfg=cfg,
        features=features,
        best_iteration=best,
        best_score=best_score,
        train_rows=kept.height,
        train_groups=int(group_sizes(kept).size),
        dropped_groups=int(dropped),
        curve=tuple(curve),
    )


def _predict_matrix(result: FitResult, X: np.ndarray) -> np.ndarray:
    return np.asarray(
        result.booster.predict(
            X, num_iteration=result.best_iteration, num_threads=result.cfg.num_threads
        ),
        dtype=np.float64,
    )


def predict(result: FitResult, frame: pl.DataFrame) -> np.ndarray:
    return _predict_matrix(result, _matrix(frame, result.features))


def gain_importance(result: FitResult) -> dict[str, dict[str, float]]:
    gains = result.booster.feature_importance("gain", iteration=result.best_iteration)
    pairs = sorted(zip(result.features, gains), key=lambda p: (-p[1], p[0]))
    by_feature = {name: float(g) for name, g in pairs}
    by_group: dict[str, float] = {}
    for group, names in feature_groups(result.features).items():
        by_group[group] = float(sum(by_feature[n] for n in names))
    by_group = dict(sorted(by_group.items(), key=lambda p: (-p[1], p[0])))
    return {"by_feature": by_feature, "by_group": by_group}


def permutation_importance(
    result: FitResult,
    frame: pl.DataFrame,
    groups: Mapping[str, Sequence[str]],
    n_pos: pl.DataFrame | None = None,
    label: str | None = None,
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    ev = RankEval(frame, label or result.cfg.label, n_pos)
    X = _matrix(frame, result.features)
    base = ev.ndcg(_predict_matrix(result, X))
    index = {f: j for j, f in enumerate(result.features)}
    out = {}
    for name, cols in groups.items():
        missing = [c for c in cols if c not in index]
        if missing or not cols:
            raise ValueError(f"group {name} has features outside the model: {missing}")
        idx = [index[c] for c in cols]
        perm = np.random.default_rng(seed).permutation(frame.height)
        saved = X[:, idx].copy()
        X[:, idx] = saved[perm]
        value = ev.ndcg(_predict_matrix(result, X))
        X[:, idx] = saved
        out[name] = {"base": base, "permuted": value, "delta": base - value}
    return out


def score_percentile(frame: pl.DataFrame, scores) -> np.ndarray:
    if frame["T"].n_unique() > 1:
        raise ValueError("percentiles need a single window")
    scored = frame.select("uid").with_columns(
        pl.Series("score", np.asarray(scores, dtype=np.float64))
    )
    return bl.user_percentile(scored, "score")


def user_folds(frame: pl.DataFrame, n_folds: int = CALIB_FOLDS, seed: int = SEED):
    uids = frame["uid"].unique().sort().to_numpy()
    fold_of = np.empty(uids.size, dtype=np.int64)
    fold_of[np.random.default_rng(seed).permutation(uids.size)] = (
        np.arange(uids.size) % n_folds
    )
    return fold_of[np.searchsorted(uids, frame["uid"].to_numpy())]


def to_probability(
    cfg: GBDTConfig,
    label: str,
    frame: pl.DataFrame,
    scores,
    calib_frame: pl.DataFrame | None = None,
    calib_scores=None,
) -> tuple[np.ndarray, str]:
    if cfg.objective == "binary" and label == cfg.label:
        return np.asarray(scores, dtype=np.float64), "binary probability"
    pct = score_percentile(frame, scores)
    if calib_frame is None:
        y = frame[label].to_numpy()
        folds = user_folds(frame)
        out = np.empty(pct.size, dtype=np.float64)
        for f in range(CALIB_FOLDS):
            mask = folds == f
            cal = bin_calibrator(pct[~mask], y[~mask], CALIB_BINS)
            out[mask] = cal.transform(pct[mask])
        return (
            out,
            f"percentile bin_calibrator cross-fitted over {CALIB_FOLDS} user folds",
        )
    cal = bin_calibrator(
        score_percentile(calib_frame, calib_scores),
        calib_frame[label].to_numpy(),
        CALIB_BINS,
    )
    return cal.transform(pct), "percentile bin_calibrator fitted on inner out-of-sample"


class ProtocolResult(NamedTuple):
    protocol: str
    target: str
    selection_window: str
    inner_train: tuple[str, ...]
    final_train: tuple[str, ...]
    inner_fit: FitResult
    fit: FitResult
    scores: dict[str, np.ndarray]
    probs: dict[str, np.ndarray]
    info: dict
    timings: dict[str, float]


def protocol_windows(cfg: GBDTConfig, protocol: str):
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    sel = check_selection_window(cfg.selection_window)
    inner = tuple(cfg.train_windows) if cfg.train_windows else INNER_TRAIN_WINDOWS
    sel_T = get_window(sel).T
    for name in inner:
        if name in FORBIDDEN_SELECTION or get_window(name).T_end > sel_T:
            raise ValueError(f"train window {name} must end before {sel} starts")
    if protocol == "inner":
        return inner, sel, inner, sel
    return inner, sel, (*inner, sel), OUTER_TARGET


def run_protocol(
    datas: Mapping[str, WindowData],
    cfg: GBDTConfig,
    protocol: str,
    features: Sequence[str] | None = None,
) -> ProtocolResult:
    inner_train, sel, final_train, target = protocol_windows(cfg, protocol)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    train = stack_frames([datas[w].frame for w in inner_train])
    valid = datas[sel].frame
    n_pos = group_n_pos(datas, [sel], cfg.label)
    inner = fit(train, valid, cfg, n_pos, features)
    del train
    timings["fit_inner"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    inner_scores = {w: predict(inner, datas[w].frame) for w in (*inner_train, sel)}
    timings["predict_inner"] = time.perf_counter() - t0
    info = {
        "inner": {
            "train_windows": list(inner_train),
            "selection_window": sel,
            "best_iteration": inner.best_iteration,
            f"best_{STOP_METRIC}": inner.best_score,
            "rounds_run": len(inner.curve),
            "train_rows": inner.train_rows,
            "train_groups": inner.train_groups,
            "dropped_zero_positive_groups": inner.dropped_groups,
        }
    }
    if protocol == "inner":
        final, scores = inner, inner_scores
        calib_frame = calib_scores = None
    else:
        rounds = round(inner.best_iteration * REFIT_FACTOR)
        t0 = time.perf_counter()
        full = stack_frames([datas[w].frame for w in final_train])
        refit_cfg = replace(cfg, num_boost_round=rounds, early_stopping_rounds=0)
        final = fit(full, None, refit_cfg, None, inner.features)
        del full
        timings["fit_outer"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = {w: predict(final, datas[w].frame) for w in (*final_train, target)}
        timings["predict_outer"] = time.perf_counter() - t0
        calib_frame, calib_scores = valid, inner_scores[sel]
        info["outer"] = {
            "train_windows": list(final_train),
            "num_boost_round": rounds,
            "refit_factor": REFIT_FACTOR,
            "train_rows": final.train_rows,
            "train_groups": final.train_groups,
            "dropped_zero_positive_groups": final.dropped_groups,
            "row_ratio_vs_inner": final.train_rows / inner.train_rows,
        }
    probs, methods = {}, {}
    for label in LABELS:
        probs[label], methods[label] = to_probability(
            cfg, label, datas[target].frame, scores[target], calib_frame, calib_scores
        )
    info["probability"] = methods
    return ProtocolResult(
        protocol=protocol,
        target=target,
        selection_window=sel,
        inner_train=inner_train,
        final_train=final_train,
        inner_fit=inner,
        fit=final,
        scores=scores,
        probs=probs,
        info=info,
        timings=timings,
    )


def eval_frame(data: WindowData) -> pl.DataFrame:
    return data.frame.select(
        "uid",
        "item_id",
        "novel",
        *LABELS,
        *(pl.col(f"{CAND_PREFIX}rank_{s}").alias(f"rank_{s}") for s in SOURCES),
    )


def add_baselines(
    events: Frame,
    data: WindowData,
    frame: pl.DataFrame,
    params: bl.BaselineParams = BASELINE_PARAMS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    T = data.window.T
    hist = history(events, T)
    base = bl.score_baselines(hist, T, frame.select("uid", "item_id"), params)
    frame = frame.with_columns(base.select(bl.BASELINES))
    frame = frame.with_columns(
        bl.user_rank_scores(hist, T, frame, params.b2_days, params.b2_variant).alias(
            "B5"
        ),
        bl.rrf_scores(frame).alias("B6"),
    )
    top = bl.catalog_popularity(
        hist, T, bl.CATALOG_TOP_N, params.b2_days, params.b2_variant
    )
    return frame, attach_labels(top, hist, data.pairs)


def evaluate_models(
    frames: dict[str, pl.DataFrame],
    n_pos: Mapping[str, Mapping[str, pl.DataFrame]],
    lists: Mapping[str, pl.DataFrame],
    n_hist: Mapping[str, int],
    model_keys: Sequence[str],
    target: str,
    calib_windows: Sequence[str],
) -> tuple[dict, dict]:
    if target in calib_windows:
        raise ValueError("baseline calibration must not use the target window")
    grid = {n: {lb: {sl: {} for sl in SLICES} for lb in LABELS} for n in frames}
    bl._calibrate(frames, SCORED_BASELINES, calib_windows, target)
    bl._evaluate_keys(frames, n_pos, [*model_keys, *SCORED_BASELINES], target, grid)
    bl._evaluate_catalog(lists, n_pos, target, grid)
    order = [*model_keys, *bl.REPORTED_BASELINES]
    metrics = {
        n: {
            lb: {sl: {m: grid[n][lb][sl][m] for m in order} for sl in SLICES}
            for lb in LABELS
        }
        for n in frames
    }
    all_users = {
        n: {
            lb: {
                sl: {
                    m: bl.all_users_metrics(metrics[n][lb][sl][m], n_hist[n])
                    for m in order
                }
                for sl in SLICES
            }
            for lb in LABELS
        }
        for n in frames
    }
    return metrics, all_users


def seed_stats(metrics: Mapping, keys: Sequence[str]) -> dict:
    out = {}
    for n, by_label in metrics.items():
        out[n] = {}
        for lb, by_slice in by_label.items():
            out[n][lb] = {}
            for sl, by_model in by_slice.items():
                names = [k for k, v in by_model[keys[0]].items() if k != "n_users"]
                stats = {}
                for m in names:
                    vals = [by_model[k][m] for k in keys]
                    if any(v is None for v in vals):
                        continue
                    arr = np.asarray(vals, dtype=np.float64)
                    stats[m] = {
                        "mean": float(arr.mean()),
                        "std": float(arr.std(ddof=1)),
                    }
                out[n][lb][sl] = stats
    return out


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return peak * scale / (1024.0 * 1024.0)


def library_versions() -> dict[str, str]:
    return {**bl.library_versions(), "lightgbm": lgb.__version__}


def metrics_digest(record: Mapping) -> str:
    stable = bl.stable_view(record, VOLATILE_KEYS)
    text = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def write_run_log(
    record: Mapping,
    name: str,
    out_dir: str | Path = "runs",
    now: datetime | None = None,
) -> tuple[Path, str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
        raise ValueError(f"invalid run name {name!r}")
    if "selection_window" not in record:
        raise ValueError("record must state selection_window")
    check_selection_window(record["selection_window"])
    now = now or datetime.now()
    run_id = f"{now:%Y%m%d-%H%M%S}-{name}"
    body = bl._clean(record)
    body["git_commit"] = GIT_COMMIT
    digest = metrics_digest(body)
    out = {
        "run_id": run_id,
        "created_at": now.isoformat(timespec="seconds"),
        **body,
        "metrics_digest": digest,
    }
    path = Path(out_dir) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return path, digest
