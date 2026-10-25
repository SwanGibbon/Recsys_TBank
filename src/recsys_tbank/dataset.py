import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from recsys_tbank import candidates as cd
from recsys_tbank import features as ft
from recsys_tbank.features import cache as fc
from recsys_tbank.split import (
    DATA_END,
    DATA_PATH,
    TEST_TS,
    VAL_TS,
    Frame,
    LeakageError,
    Window,
    get_window,
    history,
    load_events,
    novel_flag,
    positive_pairs,
)

SEED = 42
PROTOCOL_LIMITS = {"inner": VAL_TS, "outer": TEST_TS, "final": DATA_END + 1}
PROTOCOL_WINDOWS = {
    "inner": ("train_266", "train_273", "train_280"),
    "outer": ("train_266", "train_273", "train_280", "val"),
    "final": ("train_266", "train_273", "train_280", "val", "test"),
}
GATED_PROTOCOLS = ("final",)
CACHE_DIR = fc.CACHE_DIR
CANDIDATES_SOURCE = Path(cd.__file__).resolve()
CAND_PREFIX = "cand:"
PAIR = ["uid", "item_id"]
KEYS = ("T", "uid", "item_id")
GROUP_KEYS = ("T", "uid")
TARGETS = ("novel", *cd.LABELS)
RESERVED = (*KEYS, *TARGETS)


class WindowData(NamedTuple):
    window: Window
    frame: pl.DataFrame
    n_pos: dict[str, pl.DataFrame]
    n_hist_users: int
    pairs: pl.DataFrame
    features: tuple[str, ...]
    timings: dict[str, float]
    cache: dict[str, bool]


def _check_protocol(protocol: str, allow_test: bool = False) -> int:
    if protocol not in PROTOCOL_LIMITS:
        raise ValueError(f"protocol must be one of {tuple(PROTOCOL_LIMITS)}")
    if protocol in GATED_PROTOCOLS and not allow_test:
        raise PermissionError(
            f"protocol {protocol} reads the test window and requires allow_test=True"
        )
    return PROTOCOL_LIMITS[protocol]


def assert_events_before(events: Frame, limit: int) -> None:
    latest = events.lazy().select(pl.col("timestamp").max()).collect().item()
    if latest is not None and latest >= limit:
        raise LeakageError(f"events reach timestamp {latest} >= limit {limit}")


def sample_users(path: str | Path = DATA_PATH, n: int = 0, seed: int = SEED):
    if n <= 0:
        return None
    uids = (
        pl.scan_parquet(path)
        .filter(pl.col("timestamp") < VAL_TS)
        .select(pl.col("uid").cast(pl.UInt32))
        .unique()
        .sort("uid")
        .collect()["uid"]
        .to_numpy()
    )
    if n >= uids.size:
        return uids
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(uids, size=n, replace=False))


def load_protocol_events(
    protocol: str,
    path: str | Path = DATA_PATH,
    uids=None,
    allow_test: bool = False,
) -> tuple[pl.DataFrame, dict]:
    limit = _check_protocol(protocol, allow_test)
    path = Path(path)
    lazy = load_events(path).filter(pl.col("timestamp") < limit)
    if uids is not None:
        keep = pl.Series("uid", np.asarray(uids), dtype=pl.UInt32)
        lazy = lazy.filter(pl.col("uid").cast(pl.UInt32).is_in(keep.implode()))
    events = lazy.collect()
    assert_events_before(events, limit)
    root = DATA_PATH.parents[2]
    shown = path.resolve()
    info = {
        "path": str(shown.relative_to(root))
        if shown.is_relative_to(root)
        else str(shown),
        "file_rows": pl.scan_parquet(path).select(pl.len()).collect().item(),
        "file_bytes": path.stat().st_size,
        "loaded_rows": events.height,
        "loaded_users": events["uid"].n_unique(),
        "loaded_filter": f"timestamp < {limit} (protocol {protocol})",
        "users_sample": None if uids is None else len(uids),
    }
    return events, info


def candidates_digest(path: str | Path = CANDIDATES_SOURCE) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_cache_key(
    hist: Frame, T: int, cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG
) -> str:
    payload = {
        "kind": "candidates",
        "T": int(T),
        "hist": fc.hist_fingerprint(hist),
        "cand_config": cfg.config_dict(),
        "candidates_sha256": candidates_digest(),
        "polars": pl.__version__,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _write_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    os.replace(tmp, path)


def get_or_build_candidates(
    hist: Frame,
    T: int,
    cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG,
    use_cache: bool = True,
    cache_dir: str | Path = CACHE_DIR,
) -> tuple[pl.DataFrame, bool]:
    if not use_cache:
        return cd.build_candidates(hist, T, cfg), False
    path = Path(cache_dir) / f"cands-{candidate_cache_key(hist, T, cfg)}.parquet"
    if path.exists():
        return pl.read_parquet(path), True
    out = cd.build_candidates(hist, T, cfg)
    _write_atomic(out, path)
    return out, False


def _feature_cache_config(cfg: cd.CandidateConfig) -> dict:
    return {**cfg.config_dict(), "candidates_sha256": candidates_digest()}


def get_or_build_features(
    hist: Frame,
    T: int,
    cands: pl.DataFrame,
    specs: Sequence[ft.FeatureSpec],
    cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG,
    use_cache: bool = True,
    cache_dir: str | Path = CACHE_DIR,
    timings: dict[str, float] | None = None,
) -> tuple[pl.DataFrame, bool]:
    if not use_cache:
        return ft.build_features(hist, T, cands, specs, timings), False
    key = fc.cache_key(hist, T, cands, specs, _feature_cache_config(cfg))
    path = fc.cache_path(key, cache_dir)
    if path.exists():
        return pl.read_parquet(path), True
    out = ft.build_features(hist, T, cands, specs, timings)
    _write_atomic(out, path)
    return out, False


def _resolve_specs(specs) -> tuple[ft.FeatureSpec, ...]:
    specs = ft.all_specs() if specs is None else tuple(specs)
    clash = sorted({s.name for s in specs} & set(RESERVED))
    if clash:
        raise ValueError(f"feature names clash with reserved columns: {clash}")
    if any(s.name.startswith(CAND_PREFIX) for s in specs):
        raise ValueError(f"feature names must not start with {CAND_PREFIX!r}")
    return specs


def _same_keys(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    cols = [pl.col(c).cast(pl.UInt32) for c in PAIR]
    return a.select(cols).equals(b.select(cols))


def build_window_frame(
    events: Frame,
    window: str | Window,
    cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG,
    specs: Sequence[ft.FeatureSpec] | None = None,
    use_cache: bool = True,
    cache_dir: str | Path = CACHE_DIR,
    allow_test: bool = False,
) -> WindowData:
    w = get_window(window) if isinstance(window, str) else window
    if w.name == "test" and not allow_test:
        raise PermissionError("test windows are never built by dataset.py")
    specs = _resolve_specs(specs)
    names = tuple(s.name for s in specs)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    hist = history(events, w.T)
    cands, cand_hit = get_or_build_candidates(hist, w.T, cfg, use_cache, cache_dir)
    timings["candidates"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    groups: dict[str, float] = {}
    feats, feat_hit = get_or_build_features(
        hist, w.T, cands, specs, cfg, use_cache, cache_dir, groups
    )
    timings["features"] = time.perf_counter() - t0
    timings.update({f"features/{k}": v for k, v in groups.items()})
    if not _same_keys(feats, cands):
        raise ValueError("feature rows do not match candidate rows")
    t0 = time.perf_counter()
    pairs = positive_pairs(events, w.T, w.T_end, w.name == "test" and allow_test)
    labeled = cd.attach_labels(cands.select(PAIR), hist, pairs)
    if not _same_keys(labeled, cands):
        raise ValueError("label rows do not match candidate rows")
    users = hist.lazy().select(pl.col("uid").cast(pl.UInt32)).unique().collect()
    pairs_novel = novel_flag(hist, pairs)
    n_pos = {
        label: cd.n_pos_table(pairs_novel, label).join(users, on="uid", how="semi")
        for label in cd.LABELS
    }
    extra = [c for c in cands.columns if c not in PAIR]
    frame = pl.concat(
        [
            cands.select(
                pl.lit(w.T, dtype=pl.Int64).alias("T"),
                *(pl.col(c).cast(pl.UInt32) for c in PAIR),
                *(pl.col(c).alias(f"{CAND_PREFIX}{c}") for c in extra),
            ),
            feats.select(pl.col(n).cast(pl.Float32) for n in names),
            labeled.select(*cd.LABELS, "novel"),
        ],
        how="horizontal",
    )
    timings["labels"] = time.perf_counter() - t0
    return WindowData(
        window=w,
        frame=frame,
        n_pos=n_pos,
        n_hist_users=users.height,
        pairs=pairs,
        features=names,
        timings=timings,
        cache={"candidates": cand_hit, "features": feat_hit},
    )


def build_dataset(
    events: Frame,
    protocol: str,
    window_names: Sequence[str] | None = None,
    cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG,
    specs: Sequence[ft.FeatureSpec] | None = None,
    use_cache: bool = True,
    cache_dir: str | Path = CACHE_DIR,
    allow_test: bool = False,
) -> dict[str, WindowData]:
    limit = _check_protocol(protocol, allow_test)
    assert_events_before(events, limit)
    names = PROTOCOL_WINDOWS[protocol] if window_names is None else window_names
    for name in names:
        if get_window(name).T_end > limit:
            raise PermissionError(
                f"window {name} needs labels at or after {limit}, "
                f"not available in protocol {protocol}"
            )
    specs = _resolve_specs(specs)
    return {
        name: build_window_frame(
            events, name, cfg, specs, use_cache, cache_dir, allow_test
        )
        for name in names
    }


def _key_arrays(frame: pl.DataFrame) -> list[np.ndarray]:
    return [frame[c].to_numpy() for c in KEYS]


def assert_sorted_keys(frame: pl.DataFrame) -> None:
    t, u, i = _key_arrays(frame)
    if t.size < 2:
        return
    gt_t, eq_t = t[1:] > t[:-1], t[1:] == t[:-1]
    gt_u, eq_u = u[1:] > u[:-1], u[1:] == u[:-1]
    ok = gt_t | (eq_t & gt_u) | (eq_t & eq_u & (i[1:] > i[:-1]))
    if not ok.all():
        raise ValueError("frame must be strictly sorted by (T, uid, item_id)")


def stack_frames(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        raise ValueError("no frames to stack")
    out = pl.concat(list(frames), how="vertical").sort(list(KEYS))
    assert_sorted_keys(out)
    return out


def group_sizes(frame: pl.DataFrame) -> np.ndarray:
    assert_sorted_keys(frame)
    if frame.height == 0:
        return np.zeros(0, dtype=np.int64)
    t, u, _ = _key_arrays(frame)
    start = np.concatenate(([True], (t[1:] != t[:-1]) | (u[1:] != u[:-1])))
    bounds = np.append(np.flatnonzero(start), frame.height)
    return np.diff(bounds).astype(np.int64)


def group_n_pos(
    datas: Mapping[str, WindowData],
    names: Sequence[str],
    label: str,
    slice_name: str = "all",
) -> pl.DataFrame:
    parts = [
        datas[n]
        .n_pos[label]
        .select(
            pl.lit(datas[n].window.T, dtype=pl.Int64).alias("T"),
            pl.col("uid").cast(pl.UInt32),
            pl.col(slice_name).cast(pl.Int64).alias("n_pos"),
        )
        for n in names
    ]
    return pl.concat(parts).sort(list(GROUP_KEYS))
