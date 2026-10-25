import importlib
import operator
import pkgutil
import time
from collections.abc import MutableMapping, Sequence

import polars as pl

from recsys_tbank.features.prep import PAIR, Context
from recsys_tbank.features.spec import (
    FEATURE_VERSION,
    GROUPS,
    FeatureSpec,
    Group,
    all_specs,
    register_group,
)
from recsys_tbank.split import Frame, assert_no_future

__all__ = [
    "ALL_SPECS",
    "FEATURE_VERSION",
    "GROUPS",
    "LABEL_COLUMNS",
    "FeatureSpec",
    "Group",
    "all_specs",
    "build_features",
    "check_inputs",
    "register_group",
]

LABEL_COLUMNS = ("y_org", "y_all", "label", "novel")
_CORE = frozenset({"spec", "prep", "cache"})

for _module in pkgutil.iter_modules(__path__):
    if _module.name not in _CORE:
        importlib.import_module(f"{__name__}.{_module.name}")

ALL_SPECS = all_specs()


def check_inputs(hist: Frame, T: int, cands: pl.DataFrame) -> int:
    assert_no_future(hist, T)
    T = operator.index(T)
    if not isinstance(cands, pl.DataFrame):
        raise TypeError(f"cands must be a polars DataFrame, got {type(cands).__name__}")
    leaked = [c for c in LABEL_COLUMNS if c in cands.columns]
    if leaked:
        raise ValueError(f"cands must not contain label columns: {leaked}")
    missing = [c for c in PAIR if c not in cands.columns]
    if missing:
        raise ValueError(f"cands is missing key columns: {missing}")
    return T


def _resolve(specs: Sequence[FeatureSpec] | None) -> tuple[FeatureSpec, ...]:
    specs = all_specs() if specs is None else tuple(specs)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    unknown = sorted({s.group for s in specs} - set(GROUPS))
    if unknown:
        raise ValueError(f"unregistered feature groups: {unknown}")
    if any(n in PAIR for n in names):
        raise ValueError("feature names must differ from the key columns")
    return specs


def build_features(
    hist: Frame,
    T: int,
    cands: pl.DataFrame,
    specs: Sequence[FeatureSpec] | None = None,
    timings: MutableMapping[str, float] | None = None,
) -> pl.DataFrame:
    T = check_inputs(hist, T, cands)
    specs = _resolve(specs)
    start = time.perf_counter()
    frame = hist.collect() if isinstance(hist, pl.LazyFrame) else hist
    ctx = Context(frame, T, cands)
    if timings is not None:
        timings["prep"] = time.perf_counter() - start
    out = ctx.keys
    used = sorted({GROUPS[s.group] for s in specs}, key=lambda g: (g.order, g.name))
    for group in used:
        start = time.perf_counter()
        part = tuple(s for s in specs if s.group == group.name)
        keys = list(group.keys)
        table = group.build(ctx, part).select(*keys, *(s.name for s in part))
        if table.select(keys).is_duplicated().any():
            raise ValueError(f"group {group.name} returned duplicate keys")
        out = out.join(table, on=keys, how="left", maintain_order="left")
        if timings is not None:
            timings[group.name] = time.perf_counter() - start
    return out.select(*PAIR, *(pl.col(s.name).cast(pl.Float32) for s in specs))
