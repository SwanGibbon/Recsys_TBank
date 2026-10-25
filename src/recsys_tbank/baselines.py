import hashlib
import json
import math
import operator
import platform
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from recsys_tbank.candidates import (
    DEFAULT_CONFIG,
    LABELS,
    SLICES,
    SOURCES,
    CandidateConfig,
    attach_labels,
    build_candidates,
    n_pos_table,
    source_report,
)
from recsys_tbank.metrics import bin_calibrator, evaluate
from recsys_tbank.split import (
    DATA_PATH,
    DAY,
    TEST_TS,
    THRESHOLD,
    WINDOW_DAYS,
    Frame,
    assert_no_future,
    get_window,
    history,
    load_events,
    novel_flag,
    positive_pairs,
    window_summary,
)

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("org", "all")
RECENT_DAYS = (1, 7, 30)
KS = (10, 100)
BASELINES = ("B0", "B1", "B2", "B3")
EXTRA_BASELINES = ("B4", "B5", "B6")
REPORTED_BASELINES = BASELINES + EXTRA_BASELINES
CATALOG_TOP_N = 100
RRF_K = 60
ALL_USERS_METRICS = tuple(
    f"{m}@{k}" for k in KS for m in ("ndcg", "recall", "recall_cap", "hit_rate")
)
EVAL_WINDOWS = ("train_266", "train_273", "train_280", "val")
CALIB_WINDOWS = ("train_266", "train_273", "train_280")
SELECT_WINDOW = "val"
SELECT_LABEL = "y_org"
SELECT_SLICE = "all"
SELECT_METRIC = "ndcg@10"
DIGITS = 6
CALIB_BINS = 20
VOLATILE_KEYS = ("run_id", "created_at", "runtime_seconds", "metrics_digest")
SLICE_FILTERS = {
    "all": None,
    "novel": pl.col("novel") == 1,
    "repeat": pl.col("novel") == 0,
}
VARIANT_RULES = {
    "org": f"listen with is_organic == 1 and played_ratio_pct >= {THRESHOLD}",
    "all": f"listen with played_ratio_pct >= {THRESHOLD}",
}
CHOICE_RULE = (
    f"each choice maximises {SELECT_METRIC} on window {SELECT_WINDOW}, label "
    f"{SELECT_LABEL}, slice {SELECT_SLICE}, values rounded to {DIGITS} decimals, "
    "ties go to the earlier grid entry (variants org then all, W ascending); "
    "order: B2 (variant, W) first, then B3 variant with the chosen B2 as tie-break, "
    "then B1 variant; train windows and test are never used for choices"
)

_PAIR = ["uid", "item_id"]


@dataclass(frozen=True)
class BaselineParams:
    b1_variant: str = "org"
    b2_variant: str = "org"
    b2_days: int = 7
    b3_variant: str = "org"
    half_life_days: float = 14.0
    lookback_days: int = 90
    tie_eps: float = 1e-6
    seed: int = SEED

    def config_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMS = BaselineParams()


class PipelineResult(NamedTuple):
    record: dict
    frames: dict[str, pl.DataFrame]
    params: BaselineParams


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")


def _guard(hist: Frame, T: int) -> int:
    assert_no_future(hist, T)
    return operator.index(T)


def signal_events(hist: Frame, variant: str = "org") -> pl.LazyFrame:
    _check_variant(variant)
    cond = (pl.col("event_type") == "listen") & (
        pl.col("played_ratio_pct") >= THRESHOLD
    )
    if variant == "org":
        cond = cond & (pl.col("is_organic") == 1)
    return (
        hist.lazy()
        .filter(cond)
        .select(
            pl.col("uid").cast(pl.UInt32),
            pl.col("item_id").cast(pl.UInt32),
            pl.col("timestamp").cast(pl.Int64).alias("ts"),
        )
    )


def _item_counts(
    hist: Frame, T: int, variant: str, days: int | None = None
) -> pl.DataFrame:
    sig = signal_events(hist, variant)
    if days is not None:
        if days <= 0:
            raise ValueError("days must be positive")
        sig = sig.filter(pl.col("ts") >= T - days * DAY)
    return (
        sig.group_by("item_id").agg(pl.len().cast(pl.Float64).alias("score")).collect()
    )


def _align(cands: pl.DataFrame, table: pl.DataFrame, on: list[str]) -> pl.Series:
    keys = cands.select(pl.col(c).cast(pl.UInt32) for c in on)
    joined = keys.join(table, on=on, how="left", maintain_order="left")
    return joined["score"].fill_null(0.0).cast(pl.Float64).alias("score")


def random_scores(cands: pl.DataFrame, seed: int = SEED) -> pl.Series:
    rng = np.random.default_rng(seed)
    return pl.Series("score", rng.random(cands.height), dtype=pl.Float64)


def global_popularity(
    hist: Frame, T: int, cands: pl.DataFrame, variant: str = "org"
) -> pl.Series:
    T = _guard(hist, T)
    return _align(cands, _item_counts(hist, T, variant), ["item_id"])


def recent_popularity(
    hist: Frame, T: int, cands: pl.DataFrame, days: int = 7, variant: str = "org"
) -> pl.Series:
    T = _guard(hist, T)
    return _align(cands, _item_counts(hist, T, variant, days), ["item_id"])


def _pair_decay(sig: pl.DataFrame, T: int, half_life_days: float) -> pl.DataFrame:
    if sig.height == 0:
        return pl.DataFrame(
            schema={"uid": pl.UInt32, "item_id": pl.UInt32, "score": pl.Float64}
        )
    uid = sig["uid"].to_numpy()
    item = sig["item_id"].to_numpy()
    age = (T - sig["ts"].to_numpy()).astype(np.float64)
    start = np.concatenate(([True], (uid[1:] != uid[:-1]) | (item[1:] != item[:-1])))
    group = np.cumsum(start) - 1
    weights = np.power(0.5, age / (half_life_days * DAY))
    return pl.DataFrame(
        {
            "uid": uid[start],
            "item_id": item[start],
            "score": np.bincount(group, weights=weights),
        },
        schema={"uid": pl.UInt32, "item_id": pl.UInt32, "score": pl.Float64},
    )


def user_repeat(
    hist: Frame,
    T: int,
    cands: pl.DataFrame,
    variant: str = "org",
    tie_days: int = 7,
    tie_variant: str | None = None,
    half_life_days: float = 14.0,
    lookback_days: int = 90,
    tie_eps: float = 1e-6,
) -> pl.Series:
    T = _guard(hist, T)
    if half_life_days <= 0 or lookback_days <= 0:
        raise ValueError("half_life_days and lookback_days must be positive")
    sig = (
        signal_events(hist, variant)
        .filter(pl.col("ts") >= T - lookback_days * DAY)
        .sort(["uid", "item_id", "ts"])
        .collect()
    )
    decay = _align(cands, _pair_decay(sig, T, half_life_days), _PAIR).to_numpy()
    norm = _popularity_norm(hist, T, cands, tie_variant or variant, tie_days)
    return pl.Series("score", decay + tie_eps * norm, dtype=pl.Float64)


def _popularity_norm(
    hist: Frame, T: int, cands: pl.DataFrame, variant: str, days: int
) -> np.ndarray:
    tie_table = _item_counts(hist, T, variant, days)
    tie = _align(cands, tie_table, ["item_id"]).to_numpy()
    top = tie_table["score"].max() if tie_table.height else 0.0
    return np.divide(tie, top) if top else np.zeros_like(tie)


def catalog_top(
    hist: Frame, T: int, n: int = CATALOG_TOP_N, days: int = 30, variant: str = "all"
) -> pl.DataFrame:
    T = _guard(hist, T)
    if n <= 0:
        raise ValueError("n must be positive")
    return (
        _item_counts(hist, T, variant, days)
        .sort(["score", "item_id"], descending=[True, False])
        .head(n)
        .with_row_index("rank", offset=1)
        .select("item_id", pl.col("rank").cast(pl.UInt32))
    )


def catalog_popularity(
    hist: Frame, T: int, n: int = CATALOG_TOP_N, days: int = 30, variant: str = "all"
) -> pl.DataFrame:
    top = catalog_top(hist, T, n, days, variant)
    users = hist.lazy().select(pl.col("uid").cast(pl.UInt32)).unique().sort("uid")
    return (
        users.collect()
        .join(top, how="cross")
        .select(
            "uid",
            "item_id",
            (n + 1 - pl.col("rank").cast(pl.Float64)).alias("score"),
        )
    )


def user_rank_scores(
    hist: Frame,
    T: int,
    cands: pl.DataFrame,
    tie_days: int = 7,
    tie_variant: str = "org",
) -> pl.Series:
    T = _guard(hist, T)
    if "rank_user" not in cands.columns:
        raise ValueError("cands must have rank_user")
    norm = _popularity_norm(hist, T, cands, tie_variant, tie_days)
    rank = cands["rank_user"].cast(pl.Float64)
    inv = (1.0 / rank.fill_null(1.0)).to_numpy()
    score = np.where(rank.is_null().to_numpy(), norm - 1.0, inv)
    return pl.Series("score", score, dtype=pl.Float64)


def rrf_scores(
    cands: pl.DataFrame, k: int = RRF_K, sources: Sequence[str] = SOURCES
) -> pl.Series:
    if k < 0:
        raise ValueError("k must be non-negative")
    missing = [f"rank_{s}" for s in sources if f"rank_{s}" not in cands.columns]
    if missing or not sources:
        raise ValueError(f"cands must have rank columns, missing {missing}")
    parts = [
        (1.0 / (k + pl.col(f"rank_{s}").cast(pl.Float64))).fill_null(0.0)
        for s in sources
    ]
    return cands.select(pl.sum_horizontal(parts).alias("score"))["score"]


def all_users_metrics(values: Mapping, n_hist_users: int) -> dict[str, float]:
    n_users = values["n_users"]
    if n_hist_users <= 0 or n_users > n_hist_users:
        raise ValueError("n_hist_users must be positive and >= n_users")
    if n_users == 0:
        return {m: 0.0 for m in ALL_USERS_METRICS}
    return {m: values[m] * n_users / n_hist_users for m in ALL_USERS_METRICS}


def score_baselines(
    hist: Frame,
    T: int,
    cands: pl.DataFrame,
    params: BaselineParams = DEFAULT_PARAMS,
) -> pl.DataFrame:
    p = params
    return cands.select(pl.col(c).cast(pl.UInt32) for c in _PAIR).with_columns(
        random_scores(cands, p.seed).alias("B0"),
        global_popularity(hist, T, cands, p.b1_variant).alias("B1"),
        recent_popularity(hist, T, cands, p.b2_days, p.b2_variant).alias("B2"),
        user_repeat(
            hist,
            T,
            cands,
            p.b3_variant,
            p.b2_days,
            p.b2_variant,
            p.half_life_days,
            p.lookback_days,
            p.tie_eps,
        ).alias("B3"),
    )


def user_percentile(frame: pl.DataFrame, col: str) -> np.ndarray:
    parts = frame.select(
        pl.col(col).rank("average").over("uid").cast(pl.Float64).alias("r"),
        pl.len().over("uid").cast(pl.Float64).alias("n"),
    )
    return np.divide(parts["r"].to_numpy(), parts["n"].to_numpy())


def select_best(rows: Sequence[tuple[str, float]]) -> str:
    best = None
    for key, value in rows:
        if value is None or math.isnan(value):
            continue
        value = round(value, DIGITS)
        if best is None or value > best[1]:
            best = (key, value)
    if best is None:
        raise ValueError("no finite candidate to select")
    return best[0]


def _clean(obj):
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return round(value, DIGITS) if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _git_output(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def resolve_git_commit(root: str | Path = PROJECT_ROOT) -> str:
    root = Path(root).resolve()
    top = _git_output(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != root:
        return "none"
    sha = _git_output(root, "rev-parse", "--verify", "-q", "HEAD")
    if sha is None or not re.fullmatch(r"[0-9a-f]{40}", sha):
        return "none"
    dirty = _git_output(
        root, "status", "--porcelain", "--untracked-files=no", "--", "."
    )
    return f"{sha}+dirty" if dirty else sha


GIT_COMMIT = resolve_git_commit()


def stable_view(record: Mapping, volatile: Sequence[str]) -> dict:
    stable = {k: v for k, v in _clean(record).items() if k not in volatile}
    if "git_commit" in stable:
        stable["git_commit"] = "none"
    return stable


def metrics_digest(record: Mapping) -> str:
    stable = stable_view(record, VOLATILE_KEYS)
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
    now = now or datetime.now()
    run_id = f"{now:%Y%m%d-%H%M%S}-{name}"
    body = _clean(record)
    body.setdefault("git_commit", GIT_COMMIT)
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


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "polars": pl.__version__,
        "numpy": np.__version__,
        "implicit": version("implicit"),
    }


def load_dev_events(path: str | Path = DATA_PATH) -> tuple[pl.DataFrame, dict]:
    path = Path(path)
    events = load_events(path).filter(pl.col("timestamp") < TEST_TS).collect()
    root = DATA_PATH.parents[2]
    shown = path.resolve()
    info = {
        "path": str(shown.relative_to(root))
        if shown.is_relative_to(root)
        else str(shown),
        "file_rows": pl.scan_parquet(path).select(pl.len()).collect().item(),
        "file_bytes": path.stat().st_size,
        "loaded_rows": events.height,
        "loaded_filter": f"timestamp < {TEST_TS} (test window excluded)",
    }
    return events, info


def _prepare_window(events: pl.DataFrame, name: str, cfg: CandidateConfig):
    w = get_window(name)
    hist = history(events, w.T)
    cands = build_candidates(hist, w.T, cfg)
    pairs = positive_pairs(events, w.T, w.T_end)
    labeled = attach_labels(cands, hist, pairs)
    users = hist.select(pl.col("uid").cast(pl.UInt32)).unique()
    pairs_novel = novel_flag(hist, pairs)
    n_pos = {
        label: n_pos_table(pairs_novel, label).join(users, on="uid", how="semi")
        for label in LABELS
    }
    return w, hist, labeled, n_pos, pairs, users.height


def _grid_stage1(hist: Frame, T: int, cands: pl.DataFrame, seed: int) -> dict:
    cols = {"B0": random_scores(cands, seed)}
    for v in VARIANTS:
        cols[f"B1:{v}"] = global_popularity(hist, T, cands, v)
    for v in VARIANTS:
        for d in RECENT_DAYS:
            cols[f"B2:{v}:w{d}"] = recent_popularity(hist, T, cands, d, v)
    return cols


def _calibrate(frames: dict, keys: Sequence[str], calib: Sequence[str], target: str):
    for key in keys:
        probs = []
        pct_train = np.concatenate([user_percentile(frames[w], key) for w in calib])
        pct = user_percentile(frames[target], key)
        for label in LABELS:
            y = np.concatenate([frames[w][label].to_numpy() for w in calib])
            cal = bin_calibrator(pct_train, y, CALIB_BINS)
            probs.append(pl.Series(f"p:{label}:{key}", cal.transform(pct)))
        frames[target] = frames[target].with_columns(probs)


def _evaluate_keys(frames, n_pos, keys, target, grid) -> None:
    for name, labeled in frames.items():
        for label in LABELS:
            for key in keys:
                prob = f"p:{label}:{key}" if name == target else None
                cols = [
                    "uid",
                    "item_id",
                    "novel",
                    pl.col(label).alias("label"),
                    pl.col(key).alias("score"),
                ]
                if prob is not None:
                    cols.append(prob)
                res = evaluate(
                    labeled.select(cols), n_pos[name][label], SLICE_FILTERS, KS, prob
                )
                for sl, values in res.items():
                    grid[name][label][sl][key] = values


def _evaluate_catalog(lists, n_pos, target, grid) -> None:
    for name, labeled in lists.items():
        for label in LABELS:
            frame = labeled.select(
                "uid", "item_id", "novel", pl.col(label).alias("label"), "score"
            )
            res = evaluate(frame, n_pos[name][label], SLICE_FILTERS, KS)
            for sl, values in res.items():
                values["roc_auc"] = None
                if name == target:
                    values["logloss"] = None
                grid[name][label][sl]["B4"] = values


def _val_value(grid, key, label=SELECT_LABEL, metric=SELECT_METRIC) -> float:
    return grid[SELECT_WINDOW][label][SELECT_SLICE][key][metric]


def run_pipeline(
    events: pl.DataFrame,
    cfg: CandidateConfig = DEFAULT_CONFIG,
    base: BaselineParams = DEFAULT_PARAMS,
    window_names: Sequence[str] = EVAL_WINDOWS,
    calib_windows: Sequence[str] = CALIB_WINDOWS,
    data: Mapping | None = None,
) -> PipelineResult:
    if "test" in window_names or "test" in calib_windows:
        raise PermissionError("baselines are evaluated on train and val windows only")
    if SELECT_WINDOW not in window_names or not set(calib_windows) <= set(window_names):
        raise ValueError("window_names must include val and all calibration windows")
    if SELECT_WINDOW in calib_windows:
        raise ValueError("calibration must not use the selection window")
    runtime: dict[str, float] = {}
    start = time.perf_counter()
    t0 = time.perf_counter()
    summary = window_summary(events).to_dicts()
    runtime["window_summary"] = time.perf_counter() - t0
    frames, n_pos, hists, cutoffs, reports, cand_info = {}, {}, {}, {}, {}, {}
    pairs, n_hist = {}, {}
    for name in window_names:
        t0 = time.perf_counter()
        w, hist, labeled, n_pos[name], pairs[name], n_hist[name] = _prepare_window(
            events, name, cfg
        )
        runtime[f"candidates/{name}"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cands = labeled.select(_PAIR)
        cols = _grid_stage1(hist, w.T, cands, base.seed)
        frames[name] = labeled.with_columns(s.alias(k) for k, s in cols.items())
        runtime[f"scoring/{name}"] = time.perf_counter() - t0
        hists[name], cutoffs[name] = hist, w.T
        reports[name] = {
            label: source_report(labeled, n_pos[name][label], label).to_dicts()
            for label in LABELS
        }
        per_user = labeled.group_by("uid").len()["len"]
        cand_info[name] = {
            "rows": labeled.height,
            "users": per_user.len(),
            "rows_per_user_max": int(per_user.max() or 0),
            "rows_per_user_mean": float(per_user.mean() or 0.0),
        }
    grid = {n: {lb: {sl: {} for sl in SLICES} for lb in LABELS} for n in window_names}
    stage1 = list(cols)
    t0 = time.perf_counter()
    _calibrate(frames, stage1, calib_windows, SELECT_WINDOW)
    runtime["calibration"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _evaluate_keys(frames, n_pos, stage1, SELECT_WINDOW, grid)
    runtime["evaluation"] = time.perf_counter() - t0
    b2_keys = [(v, d) for v in VARIANTS for d in RECENT_DAYS]
    b2_key = select_best(
        [(f"B2:{v}:w{d}", _val_value(grid, f"B2:{v}:w{d}")) for v, d in b2_keys]
    )
    _, b2_variant, b2_w = b2_key.split(":")
    b2_days = int(b2_w[1:])
    t0 = time.perf_counter()
    b3_keys = [f"B3:{v}" for v in VARIANTS]
    for name in window_names:
        cands = frames[name].select(_PAIR)
        frames[name] = frames[name].with_columns(
            user_repeat(
                hists[name],
                cutoffs[name],
                cands,
                v,
                b2_days,
                b2_variant,
                base.half_life_days,
                base.lookback_days,
                base.tie_eps,
            ).alias(f"B3:{v}")
            for v in VARIANTS
        )
    runtime["scoring/B3"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _calibrate(frames, b3_keys, calib_windows, SELECT_WINDOW)
    runtime["calibration"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    _evaluate_keys(frames, n_pos, b3_keys, SELECT_WINDOW, grid)
    runtime["evaluation"] += time.perf_counter() - t0
    b3_key = select_best([(k, _val_value(grid, k)) for k in b3_keys])
    b1_key = select_best([(f"B1:{v}", _val_value(grid, f"B1:{v}")) for v in VARIANTS])
    chosen = {"B0": "B0", "B1": b1_key, "B2": b2_key, "B3": b3_key}
    t0 = time.perf_counter()
    lists = {}
    for name in window_names:
        hist, T = hists[name], cutoffs[name]
        cands = frames[name].select(*_PAIR, *(f"rank_{s}" for s in SOURCES))
        frames[name] = frames[name].with_columns(
            user_rank_scores(hist, T, cands, b2_days, b2_variant).alias("B5"),
            rrf_scores(cands).alias("B6"),
        )
        top = catalog_popularity(hist, T, CATALOG_TOP_N, b2_days, b2_variant)
        lists[name] = attach_labels(top, hist, pairs[name])
    runtime["scoring/B4-B6"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _calibrate(frames, ["B5", "B6"], calib_windows, SELECT_WINDOW)
    runtime["calibration"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    _evaluate_keys(frames, n_pos, ["B5", "B6"], SELECT_WINDOW, grid)
    _evaluate_catalog(lists, n_pos, SELECT_WINDOW, grid)
    runtime["evaluation"] += time.perf_counter() - t0
    report_keys = {**chosen, **{b: b for b in EXTRA_BASELINES}}
    params = BaselineParams(
        b1_variant=b1_key.split(":")[1],
        b2_variant=b2_variant,
        b2_days=b2_days,
        b3_variant=b3_key.split(":")[1],
        half_life_days=base.half_life_days,
        lookback_days=base.lookback_days,
        tie_eps=base.tie_eps,
        seed=base.seed,
    )
    metrics = {
        n: {
            lb: {
                sl: {b: grid[n][lb][sl][report_keys[b]] for b in REPORTED_BASELINES}
                for sl in SLICES
            }
            for lb in LABELS
        }
        for n in window_names
    }
    metrics_all_users = {
        n: {
            lb: {
                sl: {
                    b: all_users_metrics(metrics[n][lb][sl][b], n_hist[n])
                    for b in REPORTED_BASELINES
                }
                for sl in SLICES
            }
            for lb in LABELS
        }
        for n in window_names
    }
    w_table = [
        {
            "variant": v,
            "W": d,
            **{
                f"{lb}/{m}": _val_value(grid, f"B2:{v}:w{d}", lb, m)
                for lb in LABELS
                for m in ("ndcg@10", "recall@10", "ndcg@100", "recall@100")
            },
        }
        for v, d in b2_keys
    ]
    windows_cfg = {
        n: {"T": get_window(n).T, "T_end": get_window(n).T_end} for n in window_names
    }
    record = {
        "seed": base.seed,
        "git_commit": GIT_COMMIT,
        "config": {
            "windows": windows_cfg,
            "window_days": WINDOW_DAYS,
            "test_ts": TEST_TS,
            "threshold": THRESHOLD,
            "labels": list(LABELS),
            "slices": list(SLICES),
            "ks": list(KS),
            "candidates": cfg.config_dict(),
            "candidate_sources": list(SOURCES),
            "baseline_base_params": base.config_dict(),
            "baseline_params": params.config_dict(),
            "variants": VARIANT_RULES,
            "recent_days_grid": list(RECENT_DAYS),
            "b3_definition": "sum over own signal events in lookback of 0.5 ** ((T - ts) / half_life) + tie_eps * B2 / max(B2 over hist items)",
            "b0_definition": f"uniform random scores, numpy default_rng(seed={base.seed}) over candidate rows sorted by (uid, item_id)",
            "b4_definition": f"catalog-wide popularity: global top-{CATALOG_TOP_N} items by count of signal events of variant {b2_variant} over the last {b2_days} days of hist(T) (the chosen B2 variant and W under choice_rule), ties by item_id ascending; the same ranked list for every user with >= 1 event before T, not restricted to the candidate pool; rows = user x list item, label = window positive of the pair, novel from hist(T); metrics come from the list itself: hits = list items among the user's window positives (in a slice the list keeps only the slice's items, in list order), denominators = all window positives of the slice; cand_recall_* = recall of the list items of the slice (cand_recall_macro equals recall@100); roc_auc and logloss not applicable (null)",
            "b5_definition": f"user source rank only: 1 / rank_user for rows from the user source; other rows get B2/max(B2) - 1 in [-1, 0] with B2 = chosen variant {b2_variant} and W = {b2_days} (the B3 tie-break term), so they rank below every user-source row; remaining ties by item_id ascending",
            "b6_definition": f"unlearned reciprocal rank fusion: sum over sources {list(SOURCES)} of 1 / ({RRF_K} + rank_s), a missing rank contributes 0; ties by item_id ascending",
            "all_users_definition": f"metrics_all_users = metric * n_users / n_hist_users for {list(ALL_USERS_METRICS)}, where n_users = metric users of the slice and n_hist_users = users with >= 1 event before T; users without a positive contribute 0, exact for these metrics; roc_auc and logloss are not scaled",
            "w_table": w_table,
            "choice_rule": CHOICE_RULE,
            "chosen": chosen,
            "extra_params": {
                "catalog_top_n": CATALOG_TOP_N,
                "catalog_variant": b2_variant,
                "catalog_days": b2_days,
                "rrf_k": RRF_K,
            },
            "calibration": f"bin_calibrator(bins={CALIB_BINS}) per baseline and label on per-uid percentile ranks of score, fitted on stacked {list(calib_windows)}, applied to {SELECT_WINDOW}; logloss reported for {SELECT_WINDOW} only",
            "metric_users": "users with >= 1 window positive in the slice and >= 1 event before T",
            "roc_auc": "pooled over all candidate rows of the slice on raw score",
        },
        "data": dict(data or {}),
        "versions": library_versions(),
        "window_summary": summary,
        "candidates": cand_info,
        "candidate_recall_by_source": reports,
        "n_users": {
            n: {
                lb: {sl: grid[n][lb][sl]["B0"]["n_users"] for sl in SLICES}
                for lb in LABELS
            }
            for n in window_names
        },
        "n_hist_users": n_hist,
        "metrics": metrics,
        "metrics_all_users": metrics_all_users,
        "grid_metrics": grid,
    }
    runtime["total"] = time.perf_counter() - start
    record["runtime_seconds"] = runtime
    return PipelineResult(record, frames, params)
