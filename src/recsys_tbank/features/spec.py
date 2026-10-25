from collections.abc import Callable, Sequence
from dataclasses import dataclass

import polars as pl

FEATURE_VERSION = "phase2-s5-1"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    entity: str
    signal: str
    window: int | None = None
    agg: str = "count"
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class Group:
    name: str
    keys: tuple[str, ...]
    order: int
    build: Callable[..., pl.DataFrame]
    specs: tuple[FeatureSpec, ...]


GROUPS: dict[str, Group] = {}


def register_group(
    name: str,
    keys: Sequence[str],
    order: int,
    build: Callable[..., pl.DataFrame],
    specs: Sequence[FeatureSpec],
) -> Group:
    specs = tuple(specs)
    if name in GROUPS:
        raise ValueError(f"group {name!r} is already registered")
    if any(s.group != name for s in specs):
        raise ValueError(f"every spec of group {name!r} must carry group={name!r}")
    taken = {s.name for g in GROUPS.values() for s in g.specs}
    names = [s.name for s in specs]
    clash = taken.intersection(names)
    if clash or len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in group {name!r}: {sorted(clash)}")
    group = Group(name, tuple(keys), order, build, specs)
    GROUPS[name] = group
    return group


def all_specs() -> tuple[FeatureSpec, ...]:
    ordered = sorted(GROUPS.values(), key=lambda g: (g.order, g.name))
    return tuple(s for g in ordered for s in g.specs)


def window_suffix(window: int | None) -> str:
    return "all" if window is None else str(window)
