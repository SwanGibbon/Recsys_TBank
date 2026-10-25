import functools
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from recsys_tbank import baselines as bl
from recsys_tbank import candidates as cd
from recsys_tbank import dataset as ds
from recsys_tbank import features as ft
from recsys_tbank import gbdt as gb
from recsys_tbank import metrics as mt
from recsys_tbank import split as sp
from recsys_tbank.features import cache as fc
from recsys_tbank.features import gen
from recsys_tbank.split import (
    DATA_PATH,
    DAY,
    VAL_TS,
    WINDOW_SEC,
    Frame,
    LeakageError,
    Window,
    history,
    label_events,
    novel_flag,
    positive_pairs,
)

SEED = 42
EXTRA_DAYS = (252, 259)
WINDOW_NAMES = ("train_252", "train_259", "train_266", "train_273", "train_280")
LOAD_LIMIT = VAL_TS
SELECTION_LIMIT_DAY = 280
SELECTION_LIMIT = SELECTION_LIMIT_DAY * DAY
FORBIDDEN = ("val", "test")
GENERATED_GROUPS = gen.GENERATED_GROUPS
HAND = "hand"
GENERATED = "generated"
PARTS = (HAND, *GENERATED_GROUPS)
LABEL_COLUMNS = (*cd.LABELS, "novel")
BUILD_VERSION = "autofeat-build-1"
CACHE_DIR = fc.CACHE_DIR
MANIFEST_PREFIX = "autofeat-manifest"
SOURCE_FILES = (Path(sp.__file__).resolve(),)


class Fold(NamedTuple):
    name: str
    train: tuple[str, ...]
    valid: str
    role: str


FOLDS = (
    Fold("S1", ("train_252", "train_259"), "train_266", "selection"),
    Fold("S2", ("train_259", "train_266"), "train_273", "selection"),
    Fold("S3", ("train_266", "train_273"), "train_280", "holdout"),
)
FOLD_ROLES = ("selection", "holdout")


def autofeat_windows() -> tuple[Window, ...]:
    extra = tuple(
        Window(f"train_{d}", d * DAY, d * DAY + WINDOW_SEC) for d in EXTRA_DAYS
    )
    known = tuple(w for w in sp.windows() if w.name in WINDOW_NAMES)
    out = tuple(sorted(extra + known, key=lambda w: w.T))
    if tuple(w.name for w in out) != WINDOW_NAMES:
        raise RuntimeError(f"unexpected autofeat windows {out}")
    return out


def check_window(window: str | Window) -> Window:
    name = window if isinstance(window, str) else window.name
    if name in FORBIDDEN:
        raise PermissionError(f"window {name!r} is never built or loaded by autofeat")
    if not isinstance(window, str) and window.T_end > LOAD_LIMIT:
        raise PermissionError(
            f"window {name!r} reaches timestamp {window.T_end} > {LOAD_LIMIT} (val start)"
        )
    for w in autofeat_windows():
        if w.name == name:
            if not isinstance(window, str) and tuple(window) != tuple(w):
                raise ValueError(
                    f"window {window} differs from the autofeat window {w}"
                )
            return w
    raise KeyError(f"unknown autofeat window {name!r}; known {WINDOW_NAMES}")


def assert_selection_horizon(
    names: Sequence[str | Window], limit: int = SELECTION_LIMIT
) -> None:
    for name in names:
        w = check_window(name)
        if w.T_end > limit:
            raise LeakageError(
                f"window {w.name} labels end at day {w.T_end / DAY:g} > "
                f"day {limit / DAY:g}; not allowed at selection time"
            )


def check_folds(folds: Sequence[Fold]) -> None:
    for f in folds:
        if f.role not in FOLD_ROLES:
            raise ValueError(f"fold {f.name} has unknown role {f.role!r}")
        valid = check_window(f.valid)
        for name in f.train:
            if check_window(name).T_end > valid.T:
                raise LeakageError(
                    f"fold {f.name}: train window {name} ends after {f.valid} starts"
                )
        if f.role == "selection":
            assert_selection_horizon((*f.train, f.valid))


def selection_folds() -> tuple[Fold, ...]:
    check_folds(FOLDS)
    return FOLDS


def selection_window_names() -> tuple[str, ...]:
    names = {
        n
        for f in selection_folds()
        if f.role == "selection"
        for n in (*f.train, f.valid)
    }
    return tuple(n for n in WINDOW_NAMES if n in names)


@functools.cache
def _generated() -> tuple[ft.FeatureSpec, ...]:
    return tuple(gen.enumerate_specs())


def generated_specs(group: str | None = None) -> tuple[ft.FeatureSpec, ...]:
    if group is not None and group not in GENERATED_GROUPS:
        raise ValueError(f"group must be one of {GENERATED_GROUPS}")
    return tuple(s for s in _generated() if group is None or s.group == group)


def hand_specs() -> tuple[ft.FeatureSpec, ...]:
    return ft.all_specs()


def feature_catalog() -> pl.DataFrame:
    rows = [(s, HAND) for s in hand_specs()] + [
        (s, GENERATED) for s in generated_specs()
    ]
    names = [s.name for s, _ in rows]
    if len(set(names)) != len(names):
        raise ValueError("feature names in the catalog are not unique")
    return pl.DataFrame(
        {
            "name": names,
            "kind": [k for _, k in rows],
            "group": [s.group for s, _ in rows],
            "entity": [s.entity for s, _ in rows],
            "signal": [s.signal for s, _ in rows],
            "agg": [s.agg for s, _ in rows],
            "window": [s.window for s, _ in rows],
            "params": [list(s.params) for s, _ in rows],
        },
        schema={
            "name": pl.String,
            "kind": pl.String,
            "group": pl.String,
            "entity": pl.String,
            "signal": pl.String,
            "agg": pl.String,
            "window": pl.Int64,
            "params": pl.List(pl.Float64),
        },
    )


@contextmanager
def generated_registered():
    present = all(g in ft.GROUPS for g in GENERATED_GROUPS)
    gen.register_generated()
    try:
        yield
    finally:
        if not present:
            gen.unregister_generated()


def load_selection_events(
    path: str | Path = DATA_PATH, uids=None
) -> tuple[pl.DataFrame, dict]:
    events, info = ds.load_protocol_events("inner", path, uids)
    latest = events["timestamp"].max() if events.height else None
    if latest is not None and latest >= LOAD_LIMIT:
        raise LeakageError(f"loaded timestamp {latest} >= val start {LOAD_LIMIT}")
    info = {
        **info,
        "load_limit": LOAD_LIMIT,
        "loaded_max_ts": None if latest is None else int(latest),
    }
    return events, info


def _sha256_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _base_key(events: Frame, hist: Frame, w: Window, cfg: cd.CandidateConfig) -> str:
    payload = {
        "kind": "autofeat-base",
        "version": BUILD_VERSION,
        "window": list(w),
        "candidates": ds.candidate_cache_key(hist, w.T, cfg),
        "labels": fc.hist_fingerprint(label_events(events, w.T, w.T_end)),
        "sources_sha256": _sha256_files(SOURCE_FILES),
        "polars": pl.__version__,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def base_paths(key: str, cache_dir: str | Path = CACHE_DIR) -> dict[str, Path]:
    root = Path(cache_dir)
    return {k: root / f"af{k}-{key}.parquet" for k in ("base", "npos", "pairs")}


def build_base(
    events: Frame, hist: Frame, w: Window, cands: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    pairs = positive_pairs(events, w.T, w.T_end)
    labeled = cd.attach_labels(cands.select(ds.PAIR), hist, pairs)
    if not ds._same_keys(labeled, cands):
        raise ValueError("label rows do not match candidate rows")
    users = hist.lazy().select(pl.col("uid").cast(pl.UInt32)).unique().collect()
    pairs_novel = novel_flag(hist, pairs)
    npos = pl.concat(
        [
            cd.n_pos_table(pairs_novel, label)
            .join(users, on="uid", how="semi")
            .select(pl.lit(label).alias("label"), pl.all())
            for label in cd.LABELS
        ]
    )
    extra = [c for c in cands.columns if c not in ds.PAIR]
    base = pl.concat(
        [
            cands.select(
                pl.lit(w.T, dtype=pl.Int64).alias("T"),
                *(pl.col(c).cast(pl.UInt32) for c in ds.PAIR),
                *(pl.col(c).alias(f"{ds.CAND_PREFIX}{c}") for c in extra),
            ),
            labeled.select(*cd.LABELS, "novel"),
        ],
        how="horizontal",
    )
    return {"base": base, "npos": npos, "pairs": pairs, "n_hist_users": users.height}


def feature_path(
    hist: Frame,
    T: int,
    cands: pl.DataFrame,
    specs: Sequence[ft.FeatureSpec],
    cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG,
    cache_dir: str | Path = CACHE_DIR,
) -> Path:
    key = fc.cache_key(hist, T, cands, specs, ds._feature_cache_config(cfg))
    return fc.cache_path(key, cache_dir)


def build_part(
    hist: Frame,
    T: int,
    cands: pl.DataFrame,
    specs: Sequence[ft.FeatureSpec],
    cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG,
    use_cache: bool = True,
    cache_dir: str | Path = CACHE_DIR,
) -> dict:
    specs = tuple(specs)
    t0 = time.perf_counter()
    path = feature_path(hist, T, cands, specs, cfg, cache_dir)
    hit = use_cache and path.exists()
    stages: dict[str, float] = {}
    if not hit:
        generated = any(s.group in GENERATED_GROUPS for s in specs)
        with generated_registered() if generated else nullcontext():
            frame, _ = ds.get_or_build_features(
                hist, T, cands, specs, cfg, False, cache_dir, stages
            )
        if not ds._same_keys(frame, cands):
            raise ValueError("feature rows do not match candidate rows")
        ds._write_atomic(frame, path)
        del frame
    return {
        "file": path.name,
        "specs": len(specs),
        "hit": bool(hit),
        "seconds": time.perf_counter() - t0,
        "bytes": path.stat().st_size,
        "stages": stages,
    }


def build_window(
    events: pl.DataFrame,
    window: str | Window,
    cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG,
    hand: Sequence[ft.FeatureSpec] | None = None,
    generated: Mapping[str, Sequence[ft.FeatureSpec]] | None = None,
    use_cache: bool = True,
    cache_dir: str | Path = CACHE_DIR,
    on_part=None,
) -> dict:
    w = check_window(window)
    ds.assert_events_before(events, LOAD_LIMIT)
    hand = hand_specs() if hand is None else tuple(hand)
    if generated is None:
        generated = {g: generated_specs(g) for g in GENERATED_GROUPS}
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    hist = history(events, w.T)
    sp.assert_no_future(hist, w.T)
    cands, cand_hit = ds.get_or_build_candidates(hist, w.T, cfg, use_cache, cache_dir)
    timings["candidates"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    paths = base_paths(_base_key(events, hist, w, cfg), cache_dir)
    base_hit = use_cache and all(p.exists() for p in paths.values())
    n_hist_users = hist["uid"].n_unique()
    if not base_hit:
        built = build_base(events, hist, w, cands)
        if built["n_hist_users"] != n_hist_users:
            raise ValueError("history user count mismatch")
        for k, p in paths.items():
            ds._write_atomic(built[k], p)
        del built
    timings["labels"] = time.perf_counter() - t0
    parts: dict[str, dict] = {}
    specs_by_part = {HAND: hand, **{g: tuple(generated[g]) for g in generated}}
    for part, specs in specs_by_part.items():
        if not specs:
            continue
        parts[part] = build_part(hist, w.T, cands, specs, cfg, use_cache, cache_dir)
        timings[f"features/{part}"] = parts[part]["seconds"]
        if on_part is not None:
            on_part(w, part, parts[part])
    counts = (
        pl.scan_parquet(paths["base"])
        .select(
            pl.len().alias("rows"),
            pl.col("uid").n_unique().alias("users"),
            *(pl.col(lb).cast(pl.Int64).sum().alias(f"pos_{lb}") for lb in cd.LABELS),
            pl.col("novel").cast(pl.Int64).sum().alias("novel_rows"),
        )
        .collect()
        .row(0, named=True)
    )
    pair_counts = (
        pl.scan_parquet(paths["pairs"])
        .select(
            pl.len().alias("pairs_y_all"),
            pl.col("y_org").cast(pl.Int64).sum().alias("pairs_y_org"),
        )
        .collect()
        .row(0, named=True)
    )
    return {
        "window": w,
        "counts": {**counts, **pair_counts, "n_hist_users": int(n_hist_users)},
        "files": {
            **{k: p.name for k, p in paths.items()},
            **{part: info["file"] for part, info in parts.items()},
        },
        "columns": {part: [s.name for s in specs_by_part[part]] for part in parts},
        "parts": parts,
        "cache": {"candidates": bool(cand_hit), "labels": bool(base_hit)},
        "timings": timings,
    }


def manifest_path(users: int = 0, cache_dir: str | Path = CACHE_DIR) -> Path:
    suffix = "" if users <= 0 else f"-u{users}"
    return Path(cache_dir) / f"{MANIFEST_PREFIX}{suffix}.json"


def write_manifest(
    builds: Sequence[Mapping], users: int = 0, cache_dir: str | Path = CACHE_DIR
) -> Path:
    columns: dict[str, list[str]] = {}
    for b in builds:
        for part, names in b["columns"].items():
            if columns.setdefault(part, names) != names:
                raise ValueError(f"windows disagree on the columns of {part}")
    flat = [c for names in columns.values() for c in names]
    if len(set(flat)) != len(flat):
        raise ValueError("feature names are not unique across parts")
    body = {
        "version": BUILD_VERSION,
        "source_sha256": fc.source_digest(),
        "feature_version": ft.FEATURE_VERSION,
        "users": users,
        "columns": columns,
        "windows": {
            b["window"].name: {
                "T": b["window"].T,
                "T_end": b["window"].T_end,
                "n_hist_users": b["counts"]["n_hist_users"],
                "files": dict(b["files"]),
            }
            for b in builds
        },
    }
    path = manifest_path(users, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def read_manifest(users: int = 0, cache_dir: str | Path = CACHE_DIR) -> dict:
    path = manifest_path(users, cache_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run scripts/run_autofeat.py --stage build first"
        )
    body = json.loads(path.read_text())
    if body.get("source_sha256") != fc.source_digest():
        raise RuntimeError(
            f"{path} was built from other feature sources; rerun the build stage"
        )
    return body


def _entry(window_name: str, users: int, cache_dir) -> tuple[Window, dict, dict]:
    w = check_window(window_name)
    man = read_manifest(users, cache_dir)
    if w.name not in man["windows"]:
        raise KeyError(f"window {w.name} is not in the autofeat manifest")
    entry = man["windows"][w.name]
    if (entry["T"], entry["T_end"]) != (w.T, w.T_end):
        raise ValueError(f"manifest window {w.name} differs from {w}")
    return w, man, entry


def _resolve_columns(man: Mapping, columns) -> tuple[list[str], dict[str, str]]:
    owner = {c: part for part, names in man["columns"].items() for c in names}
    if columns is None:
        cols = [c for names in man["columns"].values() for c in names]
    else:
        cols = [columns] if isinstance(columns, str) else list(columns)
    if len(set(cols)) != len(cols):
        raise ValueError("requested columns are not unique")
    unknown = [c for c in cols if c not in owner]
    if unknown:
        raise KeyError(f"unknown feature columns: {unknown[:10]}")
    return cols, owner


def load_frame(
    window_name: str,
    columns: Sequence[str] | str | None = None,
    users: int = 0,
    cache_dir: str | Path = CACHE_DIR,
) -> pl.DataFrame:
    w, man, entry = _entry(window_name, users, cache_dir)
    cols, owner = _resolve_columns(man, columns)
    root = Path(cache_dir)
    base = pl.read_parquet(root / entry["files"]["base"])
    if base.height and (base["T"] != w.T).any():
        raise LeakageError(f"base frame of {w.name} holds rows of another window")
    keys = base.select(ds.PAIR)
    feats = []
    for part in PARTS:
        wanted = [c for c in cols if owner[c] == part]
        if not wanted:
            continue
        table = (
            pl.scan_parquet(root / entry["files"][part])
            .select(*ds.PAIR, *wanted)
            .collect()
        )
        if not ds._same_keys(table, keys):
            raise ValueError(f"{part} rows of {w.name} do not match the candidate rows")
        feats.append(table.drop(ds.PAIR))
    parts = [base.drop(LABEL_COLUMNS)]
    if feats:
        merged = pl.concat(feats, how="horizontal")
        parts.append(merged.select(pl.col(c).cast(pl.Float32) for c in cols))
    parts.append(base.select(LABEL_COLUMNS))
    return pl.concat(parts, how="horizontal")


def load_window_data(
    window_name: str,
    columns: Sequence[str] | str | None = None,
    users: int = 0,
    cache_dir: str | Path = CACHE_DIR,
) -> ds.WindowData:
    w, man, entry = _entry(window_name, users, cache_dir)
    cols, _ = _resolve_columns(man, columns)
    root = Path(cache_dir)
    frame = load_frame(w.name, cols, users, cache_dir)
    npos = pl.read_parquet(root / entry["files"]["npos"])
    n_pos = {lb: npos.filter(pl.col("label") == lb).drop("label") for lb in cd.LABELS}
    return ds.WindowData(
        window=w,
        frame=frame,
        n_pos=n_pos,
        n_hist_users=int(entry["n_hist_users"]),
        pairs=pl.read_parquet(root / entry["files"]["pairs"]),
        features=tuple(cols),
        timings={},
        cache={"candidates": True, "features": True},
    )


def reproduce_r0(
    params: Mapping[str, object],
    users: int = 0,
    cache_dir: str | Path = CACHE_DIR,
    seed: int = SEED,
    timings: MutableMapping[str, float] | None = None,
) -> dict:
    fold = next(f for f in selection_folds() if f.name == "S3")
    cfg = gb.GBDTConfig(params=dict(params), seed=seed)
    inner, sel, _, _ = gb.protocol_windows(cfg, "inner")
    if (tuple(inner), sel) != (fold.train, fold.valid):
        raise ValueError(f"gbdt inner protocol {inner}, {sel} is not fold S3")
    features = gb.resolve_features(cfg, hand_specs())
    t0 = time.perf_counter()
    datas = {
        n: load_window_data(n, features, users, cache_dir)
        for n in (*fold.train, fold.valid)
    }
    if timings is not None:
        timings["r0/load"] = time.perf_counter() - t0
    res = gb.run_protocol(datas, cfg, "inner", features)
    if timings is not None:
        timings.update({f"r0/{k}": v for k, v in res.timings.items()})
    return {
        "fold": fold._asdict(),
        "config": cfg.config_dict(),
        "n_features": len(features),
        "value": res.inner_fit.best_score,
        "inner": res.info["inner"],
    }


SCREEN_VERSION = "autofeat-screen-1"
SCREEN_LABEL = "y_org"
SCREEN_SLICE = "all"
SCREEN_K = 10
SCREEN_WINDOWS = ("train_252", "train_259", "train_266", "train_273")
SCREEN_FIT = FOLDS[0].train
HASH_WINDOWS = ("train_252", "train_266")
RHO_WINDOWS = ("train_259", "train_266")
RHO_MAX = 0.95
RHO_SHARE = 0.2
RHO_BLOCK = 128
POOL_MAX = 400
TOP_KS = (114, 40)
SCREEN_BATCH = 150
SCREEN_WORKERS = 4
SCREEN_STATUSES = ("kept", "pruned", "duplicate", "constant", "all_null", "not_reached")


def check_screen_windows(
    windows: Sequence[str] = SCREEN_WINDOWS,
    fit: Sequence[str] = SCREEN_FIT,
    hash_windows: Sequence[str] = HASH_WINDOWS,
    rho_windows: Sequence[str] = RHO_WINDOWS,
) -> tuple[str, ...]:
    windows = tuple(windows)
    assert_selection_horizon(windows)
    allowed = selection_window_names()
    if len(set(windows)) != len(windows) or not set(windows) <= set(allowed):
        raise ValueError(f"screening windows {windows} must be distinct and in {allowed}")
    for role, names in (("fit", fit), ("hash", hash_windows), ("rho", rho_windows)):
        assert_selection_horizon(names)
        if not names or not set(names) <= set(windows):
            raise ValueError(f"{role} windows {tuple(names)} must be within {windows}")
    return windows


def screen_frame(
    window_name: str,
    columns: Sequence[str],
    users: int = 0,
    cache_dir: str | Path = CACHE_DIR,
) -> pl.DataFrame:
    assert_selection_horizon([window_name])
    cols = list(columns)
    frame = load_frame(window_name, cols, users, cache_dir)
    return frame.select(*ds.PAIR, SCREEN_LABEL, *cols)


def screen_npos(
    window_name: str, users: int = 0, cache_dir: str | Path = CACHE_DIR
) -> pl.DataFrame:
    assert_selection_horizon([window_name])
    _, _, entry = _entry(window_name, users, cache_dir)
    npos = pl.read_parquet(Path(cache_dir) / entry["files"]["npos"])
    return npos.filter(pl.col("label") == SCREEN_LABEL).select(
        "uid", pl.col(SCREEN_SLICE).alias("n_pos")
    )


class RankIndex(NamedTuple):
    perm: np.ndarray
    uid: np.ndarray
    y: np.ndarray
    pos: np.ndarray
    elig: np.ndarray
    user: np.ndarray
    weight: np.ndarray
    n_users: int


def rank_index(
    frame: pl.DataFrame,
    n_pos: pl.DataFrame,
    label: str = SCREEN_LABEL,
    k: int = SCREEN_K,
) -> RankIndex:
    checked = mt._check_frames(
        frame.select(*ds.PAIR, pl.col(label).alias("label")), n_pos
    )
    keys = (
        frame.select(*ds.PAIR, label)
        .with_row_index("row")
        .sort(list(ds.PAIR), maintain_order=True)
    )
    perm = keys["row"].to_numpy().astype(np.int64)
    uid = keys["uid"].to_numpy()
    y = keys[label].to_numpy().astype(np.float64)
    users = checked.filter(pl.col("n_pos") >= 1).sort("uid")
    known = users["uid"].to_numpy()
    npos = users["n_pos"].to_numpy()
    at = np.searchsorted(known, uid)
    elig = at < known.size
    elig[elig] = known[at[elig]] == uid[elig]
    user = np.zeros(uid.size, dtype=np.uint16 if known.size < 2**16 else np.int64)
    user[elig] = at[elig]
    owner = at[elig]
    counts = np.bincount(owner, minlength=known.size)
    starts = np.cumsum(counts) - counts
    within = np.arange(owner.size) - starts[owner]
    disc = np.where(within < k, 1.0 / np.log2(within + 2.0), 0.0)
    ideal = np.concatenate(([0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))))
    idcg = ideal[np.minimum(npos, k)]
    return RankIndex(
        perm=perm,
        uid=uid,
        y=y,
        pos=y > 0,
        elig=elig,
        user=user,
        weight=disc / idcg[owner],
        n_users=int(known.size),
    )


def missing_low(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return np.where(np.isnan(v), np.float32(-np.inf), v)


class SortedColumn(NamedTuple):
    order: np.ndarray
    values: np.ndarray
    starts: np.ndarray


def sort_column(f: np.ndarray) -> SortedColumn:
    order = np.argsort(-f, kind="stable")
    v = f[order]
    starts = np.flatnonzero(np.concatenate(([True], v[1:] != v[:-1])))
    return SortedColumn(order, v, starts)


def _auc_parts(idx: RankIndex, col: SortedColumn) -> tuple[np.ndarray, np.ndarray]:
    lab = idx.pos[col.order]
    return col.values[lab], col.values[~lab][::-1]


def _u_stat(pos: np.ndarray, neg: np.ndarray) -> float:
    lo = np.searchsorted(neg, pos, "left").sum()
    hi = np.searchsorted(neg, pos, "right").sum()
    return (int(lo) + int(hi)) / 2.0


def pooled_auc(parts: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    n1 = sum(p.size for p, _ in parts)
    n0 = sum(n.size for _, n in parts)
    if n1 == 0 or n0 == 0:
        return float("nan")
    u = sum(_u_stat(p, n) for p, _ in parts for _, n in parts)
    return float(u / (n1 * n0))


def _reverse_ties(order: np.ndarray, starts: np.ndarray) -> np.ndarray:
    n = order.size
    lengths = np.diff(np.append(starts, n))
    rs, rl = starts[::-1], lengths[::-1]
    shift = np.repeat(rs - (np.cumsum(rl) - rl), rl)
    return order[shift + np.arange(n)]


def oriented_ndcg(idx: RankIndex, col: SortedColumn, direction: int) -> float:
    if idx.n_users == 0:
        return float("nan")
    order = col.order if direction > 0 else _reverse_ties(col.order, col.starts)
    kept = order[idx.elig[order]]
    final = kept[np.argsort(idx.user[kept], kind="stable")]
    return float(idx.y[final] @ idx.weight / idx.n_users)


def s1_direction(auc_fit: float) -> int:
    return 1 if auc_fit > 0.5 else -1


def rank_z(f: np.ndarray) -> np.ndarray:
    if f.size == 0:
        return np.zeros(0, dtype=np.float32)
    order = np.argsort(f, kind="stable")
    v = f[order]
    starts = np.flatnonzero(np.concatenate(([True], v[1:] != v[:-1])))
    ends = np.append(starts[1:], v.size)
    r = np.empty(v.size)
    r[order] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    r -= r.mean()
    norm = float(np.sqrt(r @ r))
    return (r / norm if norm > 0 else np.zeros_like(r)).astype(np.float32)


def column_hash(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for v in values:
        v = np.asarray(v, dtype=np.float32)
        missing = np.isnan(v)
        digest.update(missing.tobytes())
        digest.update(np.where(missing, np.float32(0), v).tobytes())
    return digest.hexdigest()


def screen_column(
    name: str,
    values: Mapping[str, np.ndarray],
    idxs: Mapping[str, RankIndex],
    fit: Sequence[str] = SCREEN_FIT,
    hash_windows: Sequence[str] = HASH_WINDOWS,
    rho_masks: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict, np.ndarray | None]:
    raw = {w: np.asarray(v, dtype=np.float32) for w, v in values.items()}
    n_rows = sum(v.size for v in raw.values())
    present = np.concatenate([v[~np.isnan(v)] for v in raw.values()])
    n_missing = n_rows - present.size
    lo = float(present.min()) if present.size else float("nan")
    hi = float(present.max()) if present.size else float("nan")
    filled = {w: missing_low(raw[w])[idxs[w].perm] for w in raw}
    cols = {w: sort_column(f) for w, f in filled.items()}
    parts = {w: _auc_parts(idxs[w], cols[w]) for w in raw}
    auc_fit = pooled_auc([parts[w] for w in fit])
    direction = s1_direction(auc_fit)
    row = {
        "name": name,
        "n_rows": n_rows,
        "missing_share": n_missing / n_rows if n_rows else float("nan"),
        "min": lo,
        "max": hi,
        "all_null": present.size == 0,
        "constant": n_missing == 0 and present.size > 0 and lo == hi,
        "hash": column_hash([raw[w] for w in hash_windows]),
        "auc_fit": auc_fit,
        "direction": direction,
        "oauc_fit": auc_fit if direction > 0 else 1.0 - auc_fit,
    }
    oauc, ndcg = [], []
    for w in raw:
        auc = pooled_auc([parts[w]])
        row[f"auc_{w}"] = auc
        row[f"oauc_{w}"] = auc if direction > 0 else 1.0 - auc
        row[f"ndcg_{w}"] = oriented_ndcg(idxs[w], cols[w], direction)
        oauc.append(row[f"oauc_{w}"])
        ndcg.append(row[f"ndcg_{w}"])
    row["ndcg_mean"] = float(np.mean(ndcg))
    row["oauc_mean"] = float(np.mean(oauc))
    z = None
    if rho_masks is not None:
        z = rank_z(np.concatenate([filled[w][m] for w, m in rho_masks.items()]))
    return row, z


def rho_sample_users(
    uids: Sequence[np.ndarray], share: float = RHO_SHARE, seed: int = SEED
) -> np.ndarray:
    pool = np.unique(np.concatenate([np.asarray(u, dtype=np.int64) for u in uids]))
    n = round(share * pool.size)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=n, replace=False))


def _batches(items: Sequence[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("batch_size must be >= 1")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def screen_features(
    features: Sequence[str],
    load: Callable[[str, Sequence[str]], pl.DataFrame],
    n_pos: Callable[[str], pl.DataFrame],
    windows: Sequence[str] = SCREEN_WINDOWS,
    fit: Sequence[str] = SCREEN_FIT,
    hash_windows: Sequence[str] = HASH_WINDOWS,
    rho_windows: Sequence[str] = RHO_WINDOWS,
    rho_share: float = RHO_SHARE,
    seed: int = SEED,
    batch_size: int = SCREEN_BATCH,
    workers: int = SCREEN_WORKERS,
    timings: MutableMapping[str, float] | None = None,
    on_batch=None,
) -> tuple[pl.DataFrame, np.ndarray, dict]:
    windows = check_screen_windows(windows, fit, hash_windows, rho_windows)
    features = list(features)
    if len(set(features)) != len(features):
        raise ValueError("screening features are not unique")
    t0 = time.perf_counter()
    idxs, keys = {}, {}
    for w in windows:
        base = load(w, [])
        idxs[w] = rank_index(base, n_pos(w))
        keys[w] = base.select(ds.PAIR)
    sample = rho_sample_users([idxs[w].uid for w in rho_windows], rho_share, seed)
    masks = {w: np.isin(idxs[w].uid, sample) for w in rho_windows}
    n_sample = int(sum(m.sum() for m in masks.values()))
    z = np.zeros((n_sample, len(features)), dtype=np.float32)
    if timings is not None:
        timings["screen/index"] = time.perf_counter() - t0
    rows: list[dict] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for b, names in enumerate(_batches(features, batch_size)):
            frames = {w: load(w, names) for w in windows}
            for w, frame in frames.items():
                if not frame.select(ds.PAIR).equals(keys[w]):
                    raise ValueError(f"rows of {w} changed between loads")

            def run(name, frames=frames):
                values = {
                    w: frames[w][name].cast(pl.Float32).fill_null(float("nan")).to_numpy()
                    for w in windows
                }
                return screen_column(name, values, idxs, fit, hash_windows, masks)

            for name, (row, col) in zip(names, pool.map(run, names)):
                z[:, len(rows)] = col
                rows.append(row)
            del frames
            if on_batch is not None:
                on_batch(b, len(rows), len(features))
    if timings is not None:
        timings["screen/columns"] = time.perf_counter() - t0
    table = pl.DataFrame(rows, infer_schema_length=None)
    info = {
        "windows": list(windows),
        "fit_windows": list(fit),
        "hash_windows": list(hash_windows),
        "rho_windows": list(rho_windows),
        "rows": {w: int(idxs[w].perm.size) for w in windows},
        "positives": {w: int(idxs[w].pos.sum()) for w in windows},
        "ndcg_users": {w: idxs[w].n_users for w in windows},
        "rho_sample": {
            "share": rho_share,
            "seed": seed,
            "users": int(sample.size),
            "rows": n_sample,
        },
    }
    return table, z, info


def greedy_prune(
    z: np.ndarray,
    order: Sequence[int],
    rho_max: float = RHO_MAX,
    cap: int = POOL_MAX,
    block: int = RHO_BLOCK,
) -> tuple[list[int], dict[int, tuple[int, float]], list[int]]:
    order = [int(j) for j in order]
    kept: list[int] = []
    dropped: dict[int, tuple[int, float]] = {}
    rest: list[int] = []
    basis = np.empty((z.shape[0], max(cap, 0)), dtype=np.float64)
    for s in range(0, len(order), max(block, 1)):
        ids = order[s : s + block]
        if len(kept) >= cap:
            rest.extend(ids)
            continue
        k0 = len(kept)
        zb = z[:, ids].astype(np.float64)
        cross = basis[:, :k0].T @ zb
        inner = zb.T @ zb
        new: list[int] = []
        for i, j in enumerate(ids):
            if len(kept) >= cap:
                rest.append(j)
                continue
            rho = np.concatenate((cross[:, i], inner[new, i]))
            if rho.size:
                m = int(np.argmax(np.abs(rho)))
                if abs(rho[m]) > rho_max:
                    dropped[j] = (kept[m], float(rho[m]))
                    continue
            new.append(i)
            kept.append(j)
        basis[:, k0 : k0 + len(new)] = zb[:, new]
    return kept, dropped, rest


def rank_order(table: pl.DataFrame) -> pl.DataFrame:
    return table.sort(
        [
            pl.col("ndcg_mean").fill_nan(None),
            pl.col("oauc_mean").fill_nan(None),
            "name",
        ],
        descending=[True, True, False],
        nulls_last=True,
    ).with_row_index("rank", offset=1)


def select_pool(
    table: pl.DataFrame,
    z: np.ndarray,
    rho_max: float = RHO_MAX,
    pool_max: int = POOL_MAX,
    block: int = RHO_BLOCK,
) -> pl.DataFrame:
    column = {n: j for j, n in enumerate(table["name"].to_list())}
    ranked = rank_order(table)
    names = ranked["name"].to_list()
    status = {}
    rep: dict[str, str | None] = {n: None for n in names}
    rho: dict[str, float | None] = {n: None for n in names}
    seen: dict[str, str] = {}
    live = []
    for n, h, const, empty in ranked.select(
        "name", "hash", "constant", "all_null"
    ).iter_rows():
        if empty:
            status[n] = "all_null"
        elif const:
            status[n] = "constant"
        elif h in seen:
            status[n], rep[n], rho[n] = "duplicate", seen[h], 1.0
        else:
            seen[h] = n
            live.append(n)
    kept, dropped, rest = greedy_prune(
        z, [column[n] for n in live], rho_max, pool_max, block
    )
    inv = {j: n for n, j in column.items()}
    pool_rank = {inv[j]: i + 1 for i, j in enumerate(kept)}
    for n in pool_rank:
        status[n] = "kept"
    for j, (r, value) in dropped.items():
        n = inv[j]
        status[n], rep[n], rho[n] = "pruned", inv[r], value
    for j in rest:
        status[inv[j]] = "not_reached"
    return ranked.with_columns(
        pl.Series("status", [status[n] for n in names], dtype=pl.String),
        pl.Series("pool_rank", [pool_rank.get(n) for n in names], dtype=pl.Int64),
        pl.Series("representative", [rep[n] for n in names], dtype=pl.String),
        pl.Series("rho", [rho[n] for n in names], dtype=pl.Float64),
    )


def pool_lists(selected: pl.DataFrame, ks: Sequence[int] = TOP_KS) -> dict[str, list[str]]:
    pool = selected.filter(pl.col("status") == "kept").sort("pool_rank")["name"].to_list()
    return {"pool": pool, **{f"top{k}": pool[:k] for k in ks}}


def list_sha256(names: Sequence[str], ordered: bool = False) -> str:
    items = list(names) if ordered else sorted(names)
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


SELECT_VERSION = "autofeat-select-1"
SELECT_LIMIT_DAY = 273 + 7
SELECT_LIMIT = SELECT_LIMIT_DAY * DAY
SELECT_LABEL = "y_org"
FIXED_GROUPS = ("C", "Z")
R3_SEEDS = (42, 43, 44)
CLUSTER_RHO = 0.90
COVER_FACTOR = 2
STABLE_MIN_FITS = 4
CLUSTER_CAP = 3
GREEDY_FOLD = "S2"
GREEDY_FAMILIES = 30
GREEDY_MAX_STEPS = 12
GREEDY_MIN_GAIN = 0.0005
GREEDY_BUDGET = 3600.0
GREEDY_SEEDS = (42, 43, 44)
PROXY_LEARNING_RATE = 0.1
PROXY_ROUNDS = 100
SELECT_METHODS = ("R2", "R3", "R4")
STOP_REASONS = ("gain_below_threshold", "max_steps", "timed_out", "no_candidates")


def select_folds() -> tuple[Fold, ...]:
    folds = tuple(f for f in selection_folds() if f.role == "selection")
    for f in folds:
        assert_selection_horizon((*f.train, f.valid), SELECT_LIMIT)
    return folds


def select_window_names(folds: Sequence[Fold] | None = None) -> tuple[str, ...]:
    folds = select_folds() if folds is None else tuple(folds)
    names = {n for f in folds for n in (*f.train, f.valid)}
    out = tuple(n for n in WINDOW_NAMES if n in names)
    assert_selection_horizon(out, SELECT_LIMIT)
    return out


def select_window_data(
    window_name: str,
    columns: Sequence[str],
    users: int = 0,
    cache_dir: str | Path = CACHE_DIR,
) -> ds.WindowData:
    assert_selection_horizon([window_name], SELECT_LIMIT)
    return load_window_data(window_name, list(columns), users, cache_dir)


def fixed_features() -> tuple[str, ...]:
    return tuple(s.name for s in hand_specs() if s.group in FIXED_GROUPS)


def select_config(
    params: Mapping[str, object], seed: int, valid: str, proxy: bool = False
) -> gb.GBDTConfig:
    if proxy:
        return gb.GBDTConfig(
            label=SELECT_LABEL,
            params={**params, "learning_rate": PROXY_LEARNING_RATE},
            seed=seed,
            num_boost_round=PROXY_ROUNDS,
            early_stopping_rounds=0,
            selection_window=valid,
        )
    return gb.GBDTConfig(
        label=SELECT_LABEL, params=dict(params), seed=seed, selection_window=valid
    )


def fold_frames(
    datas: Mapping[str, ds.WindowData],
    fold: Fold,
    columns: Sequence[str],
    cfg: gb.GBDTConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, int]:
    assert_selection_horizon((*fold.train, fold.valid), SELECT_LIMIT)
    cols = [*ds.KEYS, *columns, cfg.label]
    parts, dropped = [], 0
    for name in fold.train:
        kept, n = gb.training_rows(datas[name].frame.select(cols), cfg)
        parts.append(kept)
        dropped += n
    train = ds.stack_frames(parts)
    del parts
    valid = datas[fold.valid].frame.select(cols)
    n_pos = ds.group_n_pos(datas, [fold.valid], cfg.label)
    return train, valid, n_pos, dropped


def cluster_z(
    frames: Mapping[str, pl.DataFrame],
    n_pos: Mapping[str, pl.DataFrame],
    columns: Sequence[str],
    share: float = RHO_SHARE,
    seed: int = SEED,
    label: str = SCREEN_LABEL,
) -> tuple[np.ndarray, dict]:
    idxs = {w: rank_index(frames[w], n_pos[w], label) for w in frames}
    sample = rho_sample_users([idxs[w].uid for w in frames], share, seed)
    masks = {w: np.isin(idxs[w].uid, sample) for w in frames}
    n = int(sum(m.sum() for m in masks.values()))
    z = np.empty((n, len(columns)), dtype=np.float32)
    for j, c in enumerate(columns):
        z[:, j] = rank_z(
            np.concatenate(
                [
                    missing_low(
                        frames[w][c].cast(pl.Float32).fill_null(float("nan")).to_numpy()
                    )[idxs[w].perm][masks[w]]
                    for w in frames
                ]
            )
        )
    info = {
        "windows": list(frames),
        "share": share,
        "seed": seed,
        "users": int(sample.size),
        "rows": n,
    }
    return z, info


def rho_matrix(z: np.ndarray) -> np.ndarray:
    z64 = z.astype(np.float64)
    return z64.T @ z64


def rho_clusters(corr: np.ndarray, threshold: float = CLUSTER_RHO) -> list[list[int]]:
    n = corr.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.nonzero(np.triu(np.abs(corr) > threshold, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda m: m[0])


def gain_shares(by_feature: Mapping[str, float], pool: Sequence[str]) -> np.ndarray:
    g = np.array([float(by_feature.get(n, 0.0)) for n in pool], dtype=np.float64)
    total = float(g.sum())
    if not total > 0:
        raise ValueError("selectable features have no gain in this fit")
    return g / total


def cluster_labels(clusters: Sequence[Sequence[int]], n: int) -> np.ndarray:
    labels = np.full(n, -1, dtype=np.int64)
    for c, members in enumerate(clusters):
        labels[list(members)] = c
    if (labels < 0).any():
        raise ValueError("clusters do not cover every feature")
    return labels


def cluster_shares(shares: np.ndarray, clusters: Sequence[Sequence[int]]) -> np.ndarray:
    shares = np.atleast_2d(np.asarray(shares, dtype=np.float64))
    labels = cluster_labels(clusters, shares.shape[1])
    return np.stack(
        [np.bincount(labels, weights=row, minlength=len(clusters)) for row in shares]
    )


def counted_clusters(
    cshare: np.ndarray, sizes: np.ndarray, cover: int
) -> tuple[np.ndarray, int]:
    c = cshare.size
    order = np.lexsort((np.arange(c), -np.asarray(cshare, dtype=np.float64)))
    cum = np.cumsum(np.asarray(sizes)[order])
    m = min(int(np.searchsorted(cum, cover, side="left")) + 1, c)
    mask = np.zeros(c, dtype=bool)
    mask[order[:m]] = True
    return mask, m


def stability_select(
    shares: np.ndarray,
    clusters: Sequence[Sequence[int]],
    k: int,
    cover: int | None = None,
    min_fits: int = STABLE_MIN_FITS,
    cap: int = CLUSTER_CAP,
) -> dict:
    shares = np.atleast_2d(np.asarray(shares, dtype=np.float64))
    n = shares.shape[1]
    cover = COVER_FACTOR * k if cover is None else cover
    labels = cluster_labels(clusters, n)
    sizes = np.array([len(m) for m in clusters], dtype=np.int64)
    cs = cluster_shares(shares, clusters)
    counted, ms = [], []
    for row in cs:
        mask, m = counted_clusters(row, sizes, cover)
        counted.append(mask)
        ms.append(m)
    hits = np.sum(counted, axis=0).astype(np.int64)
    stable = hits >= min_fits
    cmean = cs.mean(axis=0)
    fmean = shares.mean(axis=0)
    order = sorted(range(len(clusters)), key=lambda c: (-cmean[c], c))
    picked: list[int] = []
    for c in order:
        if not stable[c]:
            continue
        for j in sorted(clusters[c], key=lambda j: (-fmean[j], j))[:cap]:
            if len(picked) < k:
                picked.append(int(j))
    n_stable = len(picked)
    taken: dict[int, int] = {}
    rest = sorted(
        (j for c in range(len(clusters)) if not stable[c] for j in clusters[c]),
        key=lambda j: (-fmean[j], j),
    )
    for j in rest:
        if len(picked) >= k:
            break
        c = int(labels[j])
        if taken.get(c, 0) < cap:
            picked.append(int(j))
            taken[c] = taken.get(c, 0) + 1
    return {
        "selected": picked,
        "k": k,
        "cover": cover,
        "m_per_fit": ms,
        "hits": hits.tolist(),
        "stable": [c for c in order if stable[c]],
        "n_stable_clusters": int(stable.sum()),
        "n_from_stable": n_stable,
        "n_filled": len(picked) - n_stable,
        "cluster_mean_share": cmean.tolist(),
        "member_mean_share": fmean.tolist(),
    }


def topk_indices(shares: np.ndarray, k: int) -> list[list[int]]:
    shares = np.atleast_2d(np.asarray(shares, dtype=np.float64))
    ids = np.arange(shares.shape[1])
    return [np.lexsort((ids, -row))[:k].tolist() for row in shares]


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def jaccard_summary(sets: Mapping[str, Sequence[int]], fold_of: Mapping[str, str]) -> dict:
    names = list(sets)
    table = {a: {b: jaccard(sets[a], sets[b]) for b in names} for a in names}
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    def mean(sel):
        vals = [table[a][b] for a, b in pairs if sel(a, b)]
        return float(np.mean(vals)) if vals else None

    folds = sorted(set(fold_of.values()))
    out = {
        "pairwise": table,
        "mean_all_pairs": mean(lambda a, b: True),
        "mean_cross_fold": mean(lambda a, b: fold_of[a] != fold_of[b]),
    }
    for f in folds:
        out[f"mean_within_{f}"] = mean(lambda a, b, f=f: fold_of[a] == fold_of[b] == f)
    return out


def family_key(entity: str, signal: str, agg: str) -> str:
    return f"{entity}|{signal}|{agg}"


def pool_families(
    pool: Sequence[str], catalog: pl.DataFrame
) -> dict[str, list[str]]:
    meta = {
        r["name"]: family_key(r["entity"], r["signal"], r["agg"])
        for r in catalog.select("name", "entity", "signal", "agg").iter_rows(named=True)
    }
    out: dict[str, list[str]] = {}
    for name in pool:
        out.setdefault(meta[name], []).append(name)
    return out


def greedy_forward(
    families: Mapping[str, Sequence[str]],
    base: Sequence[str],
    score: Callable[[Sequence[str], int], float],
    seeds: Sequence[int] = GREEDY_SEEDS,
    min_gain: float = GREEDY_MIN_GAIN,
    max_steps: int = GREEDY_MAX_STEPS,
    budget: float = GREEDY_BUDGET,
    clock: Callable[[], float] = time.monotonic,
    on_eval=None,
) -> dict:
    t0 = clock()
    seed = seeds[0]
    current = list(base)
    chosen: list[str] = []
    remaining = list(families)
    timings: dict = {"steps": []}

    def out_of_time():
        return clock() - t0 >= budget

    t = clock()
    current_score = float(score(current, seed))
    timings["base"] = clock() - t
    base_score = current_score
    steps: list[dict] = []
    std = threshold = None
    std_scores: list[float] = []
    stop = None
    for step in range(1, max_steps + 1):
        if not remaining:
            stop = "no_candidates"
            break
        evals, secs = [], []
        for fam in remaining:
            if out_of_time():
                stop = "timed_out"
                break
            t = clock()
            s = float(score([*current, *families[fam]], seed))
            secs.append(clock() - t)
            evals.append({"family": fam, "score": s, "gain": s - current_score})
            if on_eval is not None:
                on_eval(step, fam, s, s - current_score)
        record = {"step": step, "evaluated": len(evals), "evals": evals}
        timings["steps"].append({"evals": secs})
        if stop is not None:
            record["complete"] = False
            steps.append(record)
            break
        best = max(evals, key=lambda e: e["gain"])
        record.update(
            {
                "complete": True,
                "best_family": best["family"],
                "best_score": best["score"],
                "best_gain": best["gain"],
            }
        )
        if threshold is None:
            std_scores = [best["score"]]
            t = clock()
            for sd in seeds[1:]:
                if out_of_time():
                    stop = "timed_out"
                    break
                std_scores.append(float(score([*current, *families[best["family"]]], sd)))
            timings["steps"][-1]["std"] = clock() - t
            if stop is not None:
                record["complete"] = False
                steps.append(record)
                break
            std = float(np.std(std_scores, ddof=1))
            threshold = max(min_gain, std)
        record["threshold"] = threshold
        record["accepted"] = best["gain"] >= threshold
        steps.append(record)
        if not record["accepted"]:
            stop = "gain_below_threshold"
            break
        chosen.append(best["family"])
        current.extend(families[best["family"]])
        current_score = best["score"]
        remaining.remove(best["family"])
    if stop is None:
        stop = "max_steps"
    return {
        "chosen": chosen,
        "base_score": base_score,
        "final_score": current_score,
        "std": std,
        "std_seeds": list(seeds),
        "std_scores": std_scores,
        "threshold": threshold,
        "min_gain": min_gain,
        "max_steps": max_steps,
        "budget_seconds": budget,
        "stop_reason": stop,
        "timed_out": stop == "timed_out",
        "steps": steps,
        "timings": {**timings, "total": clock() - t0},
    }


def proxy_scorer(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    n_pos: pl.DataFrame,
    params: Mapping[str, object],
    valid_name: str,
) -> Callable[[Sequence[str], int], float]:
    ev = gb.RankEval(valid, SELECT_LABEL, n_pos)

    def score(features: Sequence[str], seed: int) -> float:
        cfg = select_config(params, seed, valid_name, proxy=True)
        res = gb.fit(train, None, cfg, None, list(features))
        return ev.ndcg(gb.predict(res, valid))

    return score


def family_list(
    chosen: Sequence[str],
    families: Mapping[str, Sequence[str]],
    mean_share: Mapping[str, float],
    pool_rank: Mapping[str, int],
    k: int,
) -> dict:
    def key(n):
        return (-mean_share[n], pool_rank[n])

    union = [n for f in chosen for n in sorted(families[f], key=key)]
    keep = set(sorted(union, key=key)[:k])
    return {
        "features": [n for n in union if n in keep],
        "union_size": len(union),
        "cut": len(union) > k,
    }


def list_record(method: str, k: int, names: Sequence[str], run_id: str) -> dict:
    if method not in SELECT_METHODS:
        raise ValueError(f"method must be one of {SELECT_METHODS}")
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("selected names are not unique")
    return {
        "method": method,
        "K": k,
        "n": len(names),
        "features": names,
        "sha256": list_sha256(names),
        "sha256_ranked": list_sha256(names, True),
        "run_id": run_id,
    }


CONFIRM_VERSION = "autofeat-confirm-1"
CONFIRM_FOLD = "S3"
CONFIRM_LIMIT = VAL_TS
CONFIRM_LABEL = "y_org"
CONFIRM_SEEDS = (42, 43, 44)
R1_SEEDS = (42,)
CONFIRM_K = 114
ELIGIBLE_METHODS = ("R2", "R3")
METHOD_COST_ORDER = ("R2", "R3", "R4")
POST_HOC_METHOD = "R3b"
NEAR_COPY_RHO = 0.98
NEAR_COPY_WINDOWS = RHO_WINDOWS
RECOVER_RHO = 0.95
HAND_GROUPS = ("U", "I", "D", "M", "X", "C", "Z")
H3_RHO = 0.95
H3_MIN_COVERED = 7
H3_HALF_LIVES = (7.0, 14.0, 30.0)
H4_MARGIN = 0.005
H5_MIN_SHARE = 0.5
BOOT_K = SCREEN_K
N_BOOT = 1000
BOOT_SEED = SEED
CI_LEVEL = 95.0
FORBIDDEN_KEY = "val"
FIXED_LABEL = "fixed_CZ"
GAIN_DIMENSIONS = ("kind", "entity", "signal", "window", "agg")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = ("src/recsys_tbank/autofeat.py",)
CODE_GLOBS = ("src/recsys_tbank/features/*.py",)
PREREG_VERSION = "autofeat-prereg-1"
PREREG_ARMS = ("G127", "G53", "U")
PREREG_SEEDS = (42, 43, 44, 45, 46)
PREREG_MARGIN = 0.005
PREREG_PRIMARY_CI = 95.0
PREREG_SECONDARY_CI = 98.3
PREREG_METRIC = "ndcg@10"
VAL_LOG_GLOB = "*-autofeat-val.json"


def confirm_fold() -> Fold:
    fold = next(f for f in selection_folds() if f.name == CONFIRM_FOLD)
    if fold.role != "holdout":
        raise ValueError(f"fold {fold.name} must be the holdout fold")
    assert_selection_horizon((*fold.train, fold.valid), CONFIRM_LIMIT)
    return fold


def confirm_window_names() -> tuple[str, ...]:
    fold = confirm_fold()
    return (*fold.train, fold.valid)


def confirm_window_data(
    window_name: str,
    columns: Sequence[str],
    users: int = 0,
    cache_dir: str | Path = CACHE_DIR,
) -> ds.WindowData:
    assert_selection_horizon([window_name], CONFIRM_LIMIT)
    return load_window_data(window_name, list(columns), users, cache_dir)


def confirm_frame(
    window_name: str,
    columns: Sequence[str],
    users: int = 0,
    cache_dir: str | Path = CACHE_DIR,
) -> pl.DataFrame:
    assert_selection_horizon([window_name], CONFIRM_LIMIT)
    return load_frame(window_name, list(columns), users, cache_dir)


def confirm_config(params: Mapping[str, object], seed: int) -> gb.GBDTConfig:
    fold = confirm_fold()
    cfg = gb.GBDTConfig(
        label=CONFIRM_LABEL, params=dict(params), seed=seed, selection_window=fold.valid
    )
    inner, sel, _, _ = gb.protocol_windows(cfg, "inner")
    if (tuple(inner), sel) != (fold.train, fold.valid):
        raise ValueError(f"gbdt inner protocol {inner}, {sel} is not fold {fold.name}")
    return cfg


def confirm_train(
    load: Callable[[str], pl.DataFrame],
    columns: Sequence[str],
    cfg: gb.GBDTConfig,
) -> tuple[pl.DataFrame, int]:
    fold = confirm_fold()
    cols = [*ds.KEYS, *columns, cfg.label]
    parts, dropped = [], 0
    for name in fold.train:
        assert_selection_horizon([name], CONFIRM_LIMIT)
        kept, n = gb.training_rows(load(name).select(cols), cfg)
        parts.append(kept)
        dropped += n
    return ds.stack_frames(parts), dropped


def fit_arm(
    train: pl.DataFrame,
    target: pl.DataFrame,
    n_pos: pl.DataFrame,
    features: Sequence[str],
    params: Mapping[str, object],
    seed: int,
) -> tuple[gb.FitResult, np.ndarray, dict]:
    cfg = confirm_config(params, seed)
    t0 = time.perf_counter()
    res = gb.fit(train, target, cfg, n_pos, list(features))
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = gb.predict(res, target)
    info = {
        "seed": seed,
        "best_iteration": res.best_iteration,
        f"best_{gb.STOP_METRIC}": res.best_score,
        "rounds_run": len(res.curve),
        "train_rows": res.train_rows,
        "train_groups": res.train_groups,
        "fit_seconds": fit_seconds,
        "predict_seconds": time.perf_counter() - t0,
    }
    return res, scores, info


def arm_metrics(
    data: ds.WindowData, key: str, scores, cfg: gb.GBDTConfig
) -> tuple[dict, dict]:
    name = data.window.name
    scores = np.asarray(scores, dtype=np.float64)
    frame = data.frame.select("uid", "item_id", "novel", *cd.LABELS).with_columns(
        pl.Series(key, scores)
    )
    probs = [
        pl.Series(f"p:{lb}:{key}", gb.to_probability(cfg, lb, data.frame, scores)[0])
        for lb in cd.LABELS
    ]
    frame = frame.with_columns(probs)
    grid = {name: {lb: {sl: {} for sl in cd.SLICES} for lb in cd.LABELS}}
    bl._evaluate_keys({name: frame}, {name: data.n_pos}, [key], name, grid)
    metrics = {lb: {sl: grid[name][lb][sl][key] for sl in cd.SLICES} for lb in cd.LABELS}
    all_users = {
        lb: {
            sl: bl.all_users_metrics(metrics[lb][sl], data.n_hist_users)
            for sl in cd.SLICES
        }
        for lb in cd.LABELS
    }
    return metrics, all_users


def per_user_ndcg(
    frame: pl.DataFrame,
    scores,
    n_pos: pl.DataFrame,
    label: str = CONFIRM_LABEL,
    slice_name: str = "all",
    k: int = BOOT_K,
) -> tuple[np.ndarray, np.ndarray]:
    scored = frame.select(
        "uid", "item_id", "novel", pl.col(label).alias("label")
    ).with_columns(pl.Series("score", np.asarray(scores, dtype=np.float64)))
    row_filter = bl.SLICE_FILTERS[slice_name]
    if row_filter is not None:
        scored = scored.filter(row_filter)
    scored = scored.select("uid", "item_id", "label", "score")
    if scored.select(pl.col("score").is_null().any() | pl.col("score").is_nan().any()).item():
        raise ValueError("score has null or NaN")
    table = mt._check_frames(scored, n_pos.select("uid", pl.col(slice_name).alias("n_pos")))
    users = table.filter(pl.col("n_pos") >= 1).sort("uid")
    ranked = (
        scored.join(users.select("uid"), on="uid", how="semi")
        .sort(["uid", "score", "item_id"], descending=[False, True, False])
        .with_columns(rank=pl.int_range(1, pl.len() + 1).over("uid"))
    )
    gain = pl.col("label").cast(pl.Float64) * (pl.col("rank") <= k)
    per_user = (
        users.join(
            ranked.group_by("uid").agg(
                (gain / (pl.col("rank") + 1).log(2)).sum().alias("dcg")
            ),
            on="uid",
            how="left",
        )
        .fill_null(0)
        .sort("uid")
    )
    ideal = np.concatenate(([0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))))
    capped = np.minimum(per_user["n_pos"].to_numpy(), k)
    dcg = per_user["dcg"].to_numpy()
    return per_user["uid"].to_numpy(), dcg / np.where(capped > 0, ideal[capped], 1.0)


def paired_bootstrap(
    a,
    b,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    ci_level: float = CI_LEVEL,
) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired vectors must be 1-d and of equal length")
    out = {"n_users": int(a.size), "n_boot": n_boot, "seed": seed, "ci_level": ci_level}
    if a.size == 0:
        return {
            **out,
            **dict.fromkeys(
                ("mean_a", "mean_b", "mean_delta", "boot_std", "ci_low", "ci_high")
            ),
            "ci_excludes_zero": None,
        }
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, d.size, d.size)].mean()
    tail = (100.0 - ci_level) / 2.0
    lo, hi = np.percentile(boots, [tail, 100.0 - tail])
    return {
        **out,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_delta": float(d.mean()),
        "boot_std": float(boots.std(ddof=1)) if n_boot > 1 else None,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def choose_method(
    scores: Mapping[str, Sequence[float]],
    eligible: Sequence[str] = ELIGIBLE_METHODS,
    cost_order: Sequence[str] = METHOD_COST_ORDER,
    digits: int = bl.DIGITS,
) -> dict:
    cand = [m for m in cost_order if m in eligible and m in scores]
    if not cand:
        raise ValueError(f"no eligible method among {sorted(scores)}")
    rounded = {m: [round(float(v), digits) for v in scores[m]] for m in cand}
    means = {m: float(np.mean(rounded[m])) for m in cand}
    stds = {
        m: float(np.std(rounded[m], ddof=1)) if len(rounded[m]) > 1 else 0.0
        for m in cand
    }
    best = max(cand, key=lambda m: (means[m], -cand.index(m)))
    tolerance = max(stds.values())
    tied = [m for m in cand if means[best] - means[m] <= tolerance]
    chosen = tied[0]
    if chosen == best:
        reason = "highest mean"
    else:
        reason = (
            f"{best} has the highest mean but {chosen} is within the larger seed std "
            f"({tolerance:.6f}) and is cheaper"
        )
    return {
        "chosen": chosen,
        "highest_mean": best,
        "candidates": cand,
        "excluded": sorted(set(scores) - set(cand)),
        "seed_scores": rounded,
        "means": means,
        "stds": stds,
        "tolerance": tolerance,
        "within_tolerance": tied,
        "difference_to_highest": {m: means[best] - means[m] for m in cand},
        "reason": reason,
    }


def near_copies(
    corr: np.ndarray,
    generated: Sequence[str],
    hand: Sequence[str],
    threshold: float = NEAR_COPY_RHO,
) -> dict[str, dict]:
    corr = np.asarray(corr, dtype=np.float64)
    if corr.shape != (len(generated), len(hand)):
        raise ValueError("corr must have shape (len(generated), len(hand))")
    out = {}
    if not len(hand):
        return out
    best = np.argmax(np.abs(corr), axis=1)
    for i, g in enumerate(generated):
        rho = float(corr[i, best[i]])
        if abs(rho) > threshold:
            out[g] = {"hand": hand[best[i]], "rho": rho}
    return out


def union_features(
    hand: Sequence[str], winning: Sequence[str], copies: Mapping[str, object]
) -> list[str]:
    out = [*hand, *(g for g in winning if g not in copies)]
    if len(set(out)) != len(out):
        raise ValueError("union feature names are not unique")
    return out


def clusters_from_members(
    pool: Sequence[str], members: Mapping[str, Sequence[str]]
) -> list[list[int]]:
    index = {n: j for j, n in enumerate(pool)}
    owner: dict[str, str] = {}
    for head, names in members.items():
        for n in names:
            if n in owner:
                raise ValueError(f"{n} is in two clusters")
            owner[n] = head
    groups: dict[str, list[int]] = {}
    for n in pool:
        groups.setdefault(owner.get(n, n), []).append(index[n])
    unknown = set(owner) - set(index)
    if unknown:
        raise KeyError(f"cluster members outside the pool: {sorted(unknown)[:5]}")
    return sorted(groups.values(), key=lambda m: m[0])


def shares_from_gain_table(gain_share: Mapping[str, Mapping], pool: Sequence[str]) -> np.ndarray:
    rows = [list(gain_share[n]["shares"]) for n in pool]
    if len({len(r) for r in rows}) != 1:
        raise ValueError("gain shares have different fit counts")
    return np.asarray(rows, dtype=np.float64).T


def r3b_select(shares: np.ndarray, clusters: Sequence[Sequence[int]], k: int) -> dict:
    n = np.atleast_2d(np.asarray(shares)).shape[1]
    return stability_select(shares, clusters, k, cap=n)


def post_hoc_record(method: str, k: int, names: Sequence[str], run_id: str, rule: str) -> dict:
    if method in SELECT_METHODS:
        raise ValueError(f"{method} is a pre-registered method, not post-hoc")
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("selected names are not unique")
    return {
        "method": method,
        "K": k,
        "n": len(names),
        "features": names,
        "sha256": list_sha256(names),
        "sha256_ranked": list_sha256(names, True),
        "run_id": run_id,
        "post_hoc": True,
        "eligible": False,
        "rule": rule,
    }


def gain_breakdown(
    by_feature: Mapping[str, float], catalog: pl.DataFrame, top: int = 15
) -> dict:
    meta = {r["name"]: r for r in catalog.iter_rows(named=True)}
    unknown = [n for n in by_feature if n not in meta]
    if unknown:
        raise KeyError(f"features outside the catalog: {unknown[:5]}")
    total = float(sum(by_feature.values()))
    out: dict = {"total_gain": total}
    for dim in GAIN_DIMENSIONS:
        acc: dict[str, float] = {}
        for n, g in by_feature.items():
            r = meta[n]
            if dim == "kind":
                label = r["kind"]
            elif r["kind"] == HAND:
                label = FIXED_LABEL
            elif dim == "window":
                label = "all" if r["window"] is None else f"w{r['window']}"
            else:
                label = str(r[dim])
            acc[label] = acc.get(label, 0.0) + float(g)
        out[f"by_{dim}"] = {
            k: (v / total if total > 0 else 0.0)
            for k, v in sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
        }
    ranked = sorted(by_feature.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    out["top"] = [
        {"name": n, "share": float(g) / total if total > 0 else 0.0} for n, g in ranked
    ]
    return out


def max_abs_rho(
    corr: np.ndarray, rows: Sequence[str], cols: Sequence[str]
) -> dict[str, dict]:
    corr = np.asarray(corr, dtype=np.float64)
    if corr.shape != (len(rows), len(cols)):
        raise ValueError("corr must have shape (len(rows), len(cols))")
    if not len(cols):
        return {r: {"best": None, "rho": None, "abs_rho": 0.0} for r in rows}
    best = np.argmax(np.abs(corr), axis=1)
    return {
        r: {
            "best": cols[best[i]],
            "rho": float(corr[i, best[i]]),
            "abs_rho": float(abs(corr[i, best[i]])),
        }
        for i, r in enumerate(rows)
    }


def group_recovery(
    best: Mapping[str, Mapping],
    group_of: Mapping[str, str],
    threshold: float = RECOVER_RHO,
    groups: Sequence[str] = HAND_GROUPS,
) -> dict:
    out = {}
    for g in groups:
        names = [n for n in best if group_of[n] == g]
        hit = [n for n in names if best[n]["abs_rho"] > threshold]
        out[g] = {
            "n_features": len(names),
            "n_recovered": len(hit),
            "recovered": bool(hit),
            "max_abs_rho": max((best[n]["abs_rho"] for n in names), default=None),
            "recovered_features": hit,
            "not_recovered": [n for n in names if n not in hit],
        }
    return out


def _half_life(params) -> float | None:
    return float(params[0]) if params else None


def h3_check(
    top10: Sequence[str],
    exact_twins: Mapping[str, str],
    best: Mapping[str, Mapping],
    g127: Sequence[str],
    catalog: pl.DataFrame,
    threshold: float = H3_RHO,
    min_covered: int = H3_MIN_COVERED,
    half_lives: Sequence[float] = H3_HALF_LIVES,
) -> dict:
    rows = []
    for h in top10:
        twin = exact_twins.get(h)
        rho = best[h]["abs_rho"] if h in best else None
        rows.append(
            {
                "hand": h,
                "exact_twin": twin,
                "best_generated": best[h]["best"] if h in best else None,
                "abs_rho": rho,
                "covered": twin is not None or (rho is not None and rho >= threshold),
            }
        )
    meta = {r["name"]: r for r in catalog.iter_rows(named=True)}
    decay = [
        n
        for n in g127
        if n in meta
        and meta[n]["kind"] == GENERATED
        and meta[n]["entity"] == "pair"
        and meta[n]["agg"] == "decay"
        and _half_life(meta[n]["params"]) in set(half_lives)
    ]
    covered = sum(r["covered"] for r in rows)
    return {
        "n_covered": covered,
        "min_covered": min_covered,
        "rho_threshold": threshold,
        "covered_part_holds": covered >= min_covered,
        "pair_decay_in_g127": decay,
        "decay_part_holds": bool(decay),
        "holds": covered >= min_covered and bool(decay),
        "top10": rows,
    }


def h4_check(
    g127: Sequence[float], g53: Sequence[float], margin: float = H4_MARGIN
) -> dict:
    a, b = float(np.mean(g127)), float(np.mean(g53))
    return {
        "g127_mean": a,
        "g53_mean": b,
        "difference": a - b,
        "margin": margin,
        "holds": a - b <= margin,
    }


def h5_check(breakdown: Mapping, min_share: float = H5_MIN_SHARE) -> dict:
    share = float(breakdown["by_entity"].get("pair", 0.0))
    generated = float(breakdown["by_kind"].get(GENERATED, 0.0))
    return {
        "pair_share_of_total_gain": share,
        "pair_share_of_generated_gain": share / generated if generated > 0 else None,
        "min_share": min_share,
        "holds": share >= min_share,
    }


def bit_equal(a, b) -> bool:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return False
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return False
    return bool(np.array_equal(a[~na].view(np.uint32), b[~nb].view(np.uint32)))


def keys_containing(obj, word: str = FORBIDDEN_KEY, path: str = "") -> list[str]:
    found = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            here = f"{path}/{k}"
            if word in str(k).lower():
                found.append(here)
            found.extend(keys_containing(v, word, here))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(keys_containing(v, word, f"{path}[{i}]"))
    return found


def assert_no_key(obj, word: str = FORBIDDEN_KEY) -> None:
    found = keys_containing(obj, word)
    if found:
        raise ValueError(f"keys containing {word!r}: {found[:10]}")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root: str | Path | None = None) -> dict[str, str]:
    root = PROJECT_ROOT if root is None else Path(root)
    paths = [root / p for p in CODE_FILES]
    for pattern in CODE_GLOBS:
        paths.extend(sorted(root.glob(pattern)))
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in paths}


def binding_digest(binding: Mapping) -> str:
    text = json.dumps(bl._clean(binding), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def prereg_arm(
    role: str,
    features: Sequence[str],
    generated: Sequence[str],
    best_iteration: int,
    list_file: str | None = None,
    derived_from: str | None = None,
    info_best_iterations: Mapping[str, int] | None = None,
) -> dict:
    features = list(features)
    if len(set(features)) != len(features):
        raise ValueError("arm features are not unique")
    return {
        "role": role,
        "n_features": len(features),
        "features": features,
        "sha256": list_sha256(features),
        "generated": list(generated),
        "generated_sha256": list_sha256(generated),
        "list_file": list_file,
        "derived_from": derived_from,
        "best_iteration_s42": int(best_iteration),
        "num_boost_round": round(int(best_iteration) * gb.REFIT_FACTOR),
        "best_iteration_other_seeds": dict(info_best_iterations or {}),
    }


PREREG_STATEMENT = (
    "The val stage (scripts/run_autofeat.py --stage val, not implemented at registration "
    "time) must call autofeat.verify_prereg on this file first and refuse to run if any "
    "recomputed hash differs (arm feature lists, list files, frozen params file, code "
    "identity of autofeat.py and features/*.py, the binding digest, the file digest, the "
    "S3 evidence file) or if any *-autofeat-val.json already exists in the runs directory. "
    "It runs once; the arms, seeds, metric, round counts, bootstrap and margins below are "
    "fixed and may not be changed after seeing val numbers."
)


def prereg_binding(
    method: str,
    arms: Mapping[str, Mapping],
    reference: Mapping,
    params_file: str,
    params_sha256: str,
    root: str | Path | None = None,
) -> dict:
    if set(arms) != set(PREREG_ARMS):
        raise ValueError(f"arms must be {PREREG_ARMS}")
    if method not in ELIGIBLE_METHODS:
        raise ValueError(f"method must be one of {ELIGIBLE_METHODS}")
    return {
        "version": PREREG_VERSION,
        "selected_method": method,
        "K": {"G127": CONFIRM_K, "G53": TOP_KS[1]},
        "arms": {a: dict(arms[a]) for a in PREREG_ARMS},
        "reference": dict(reference),
        "params": {"file": params_file, "sha256": params_sha256},
        "objective": "lambdarank",
        "label": CONFIRM_LABEL,
        "rounds_rule": (
            f"num_boost_round = round(best_iteration x {gb.REFIT_FACTOR}) with "
            "best_iteration of the S3 fit (train_266 + train_273, early stopping on "
            "train_280) of the same arm for seed 42; every val seed of the arm uses this "
            "round count; refit on train_266 + train_273 + train_280 without early stopping"
        ),
        "round_factor": gb.REFIT_FACTOR,
        "val_seeds": list(PREREG_SEEDS),
        "primary": {
            "metric": PREREG_METRIC,
            "label": CONFIRM_LABEL,
            "slice": "all",
            "split": "val",
            "n_pos": "all window positives",
            "per_user": "per-user ndcg@10 averaged over the val seeds",
            "comparison": "G127 minus E1a",
        },
        "bootstrap": {
            "kind": "paired user bootstrap",
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
            "rng": "numpy default_rng(42) restarted for every comparison",
            "users": "users with at least one positive of the slice",
        },
        "hypotheses": {
            "H1": {
                "arm": "G127",
                "statement": "non-inferiority: lower bound of the 95% CI of "
                "ndcg@10(G127) - ndcg@10(E1a) > -0.005",
                "ci_level": PREREG_PRIMARY_CI,
                "margin": PREREG_MARGIN,
            },
            "H2": {
                "arm": "U",
                "statement": "superiority: lower bound of the 98.3% CI of "
                "ndcg@10(U) - ndcg@10(E1a) > 0",
                "ci_level": PREREG_SECONDARY_CI,
            },
            "secondary": {
                "arms": ["G53", "U"],
                "ci_level": PREREG_SECONDARY_CI,
                "note": "Bonferroni over the three val comparisons",
            },
        },
        "code_identity": code_identity(root),
    }


def prereg_record(binding: Mapping, evidence: Mapping) -> dict:
    return {
        "seed": SEED,
        "seeds": list(PREREG_SEEDS),
        "git_commit": gb.GIT_COMMIT,
        "stage": "prereg",
        "protocol": "inner",
        "selection_window": confirm_fold().valid,
        "binding": dict(binding),
        "decision_digest": binding_digest(binding),
        "evidence": dict(evidence),
        "statement": PREREG_STATEMENT,
    }


def verify_prereg(path: str | Path, root: str | Path | None = None) -> dict:
    path = Path(path)
    root = PROJECT_ROOT if root is None else Path(root)
    body = json.loads(path.read_text())
    diff: list[str] = []
    if body.get("metrics_digest") != gb.metrics_digest(body):
        diff.append("file digest differs from the stored metrics_digest")
    binding = body.get("binding", {})
    if body.get("decision_digest") != binding_digest(binding):
        diff.append("binding digest differs from the stored decision_digest")
    for name, arm in binding.get("arms", {}).items():
        if list_sha256(arm["features"]) != arm["sha256"]:
            diff.append(f"arm {name}: feature list sha256 differs")
        if list_sha256(arm["generated"]) != arm["generated_sha256"]:
            diff.append(f"arm {name}: generated list sha256 differs")
        if arm.get("list_file"):
            lf = root / arm["list_file"]
            if not lf.is_file():
                diff.append(f"arm {name}: list file {arm['list_file']} is missing")
            elif list_sha256(json.loads(lf.read_text())["features"]) != arm["generated_sha256"]:
                diff.append(f"arm {name}: list file {arm['list_file']} differs")
    ref = binding.get("reference", {})
    if ref and list_sha256(ref.get("features", [])) != ref.get("sha256"):
        diff.append("reference arm feature list sha256 differs")
    params = binding.get("params", {})
    pf = root / params.get("file", "")
    if not pf.is_file():
        diff.append(f"params file {params.get('file')} is missing")
    elif file_sha256(pf) != params.get("sha256"):
        diff.append(f"params file {params.get('file')} sha256 differs")
    stored = binding.get("code_identity", {})
    current = code_identity(root)
    for f in sorted(set(stored) | set(current)):
        if stored.get(f) != current.get(f):
            diff.append(f"code identity differs: {f}")
    ev = body.get("evidence", {})
    if ev.get("file"):
        ef = path.parent / ev["file"]
        if not ef.is_file():
            diff.append(f"evidence file {ev['file']} is missing")
        elif file_sha256(ef) != ev.get("sha256"):
            diff.append(f"evidence file {ev['file']} sha256 differs")
    for p in sorted(path.parent.glob(VAL_LOG_GLOB)):
        diff.append(f"{p.name} already exists")
    return {"ok": not diff, "diff": diff, "file": path.name}
