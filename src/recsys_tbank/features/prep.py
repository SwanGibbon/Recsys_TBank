from collections.abc import Callable, Mapping, Sequence

import numpy as np
import polars as pl

from recsys_tbank.features.spec import FeatureSpec, window_suffix
from recsys_tbank.split import (
    BUCKET_SEC,
    DAY,
    THRESHOLD,
    burst_flags,
    net_like_state,
)

WINDOWS = (1, 7, 30, 90, None)
SMOOTH_M = 20.0
SKIP_PCT = 10
PAIR = ["uid", "item_id"]

Rule = Callable[[pl.DataFrame, FeatureSpec], np.ndarray | pl.Series]


def in_window(T: int, window: int | None) -> pl.Expr:
    if window is None:
        return pl.col("ts").is_not_null()
    return pl.col("ts") >= T - window * DAY


class Context:
    def __init__(self, hist: pl.DataFrame, T: int, cands: pl.DataFrame):
        self.hist = hist
        self.T = T
        self.cands = cands
        self.keys = cands.select(pl.col(c).cast(pl.UInt32) for c in PAIR)
        self._memo: dict[str, object] = {}
        ts = pl.col("timestamp").cast(pl.Int64)
        self.events = hist.select(
            pl.col("uid").cast(pl.UInt32),
            pl.col("item_id").cast(pl.UInt32),
            ts.alias("ts"),
            (ts // DAY).alias("day"),
            pl.col("event_type").cast(pl.String),
        )
        ratio = pl.col("played_ratio_pct").cast(pl.Int64)
        self.listens = hist.filter(pl.col("event_type") == "listen").select(
            pl.col("uid").cast(pl.UInt32),
            pl.col("item_id").cast(pl.UInt32),
            ts.alias("ts"),
            (ts // DAY).alias("day"),
            ratio.alias("ratio"),
            (pl.col("is_organic") == 1).alias("org"),
            (ratio >= THRESHOLD).alias("pos"),
            ((ratio >= THRESHOLD) & (pl.col("is_organic") == 1)).alias("orgpos"),
            (ratio < SKIP_PCT).alias("skip"),
            pl.col("track_length_seconds").cast(pl.Int64).alias("tl"),
        )

    def memo(self, name: str, fn: Callable[[], object]):
        if name not in self._memo:
            self._memo[name] = fn()
        return self._memo[name]

    def pairs_all(self) -> pl.DataFrame:
        return self.memo(
            "pairs_all",
            lambda: self.listens.group_by(PAIR).agg(
                pl.len().alias("n"), pl.col("ts").min().alias("first")
            ),
        )

    def item_listens(self, window: int | None) -> pl.DataFrame:
        return self.memo(
            f"item_lis_{window_suffix(window)}",
            lambda: (
                self.listens.filter(in_window(self.T, window))
                .group_by("item_id")
                .agg(pl.len().alias("n"))
            ),
        )

    def user_listens(self, window: int | None) -> pl.DataFrame:
        return self.memo(
            f"user_lis_{window_suffix(window)}",
            lambda: (
                self.listens.filter(in_window(self.T, window))
                .group_by("uid")
                .agg(pl.len().alias("n"))
            ),
        )

    def likes(self) -> pl.DataFrame:
        def make() -> pl.DataFrame:
            flags = burst_flags(self.hist).select(
                pl.col("uid").cast(pl.UInt32), "bucket", "is_burst"
            )
            return (
                self.events.filter(pl.col("event_type") == "like")
                .with_columns((pl.col("ts") // BUCKET_SEC).alias("bucket"))
                .join(flags, on=["uid", "bucket"], how="left")
                .with_columns(pl.col("is_burst").fill_null(0).cast(pl.Int64))
            )

        return self.memo("likes", make)

    def like_state(self) -> pl.DataFrame:
        return self.memo(
            "like_state",
            lambda: net_like_state(self.hist).select(
                pl.col("uid").cast(pl.UInt32),
                pl.col("item_id").cast(pl.UInt32),
                pl.col("liked").cast(pl.Int64),
            ),
        )


def column(frame: pl.DataFrame, name: str) -> np.ndarray:
    return frame[name].cast(pl.Float64).to_numpy()


def ratio(num, den) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(num.shape, np.nan)
    ok = (den != 0) & ~np.isnan(den) & ~np.isnan(num)
    np.divide(num, den, out=out, where=ok)
    return out


def global_rate(k, n) -> float:
    k = np.nan_to_num(np.asarray(k, dtype=np.float64))
    n = np.nan_to_num(np.asarray(n, dtype=np.float64))
    total = n.sum()
    return k.sum() / total if total else np.nan


def smoothed(k, n, prior: float, m: float = SMOOTH_M) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    return np.divide(k + m * prior, n + m)


def segments(frame: pl.DataFrame, keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    start = np.zeros(frame.height, dtype=bool)
    if frame.height:
        start[0] = True
        for key in keys:
            col = frame[key].to_numpy()
            start[1:] |= col[1:] != col[:-1]
    return np.cumsum(start) - 1, start


def ts_counts(frame: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    keys = list(keys)
    return frame.group_by([*keys, "ts"]).agg(pl.len().alias("_n")).sort([*keys, "ts"])


def decay(
    counts: pl.DataFrame,
    keys: Sequence[str],
    T: int,
    half_life_days: float,
    window: int | None = None,
) -> pl.DataFrame:
    keys = list(keys)
    if window is not None:
        counts = counts.filter(pl.col("ts") >= T - window * DAY)
    group, start = segments(counts, keys)
    age = (T - counts["ts"].to_numpy()).astype(np.float64)
    weights = counts["_n"].to_numpy().astype(np.float64) * np.power(
        0.5, age / (half_life_days * DAY)
    )
    out = counts.filter(pl.Series(start)).select(keys)
    values = np.bincount(group, weights=weights, minlength=out.height)
    return out.with_columns(pl.Series("value", values, dtype=pl.Float64))


def aligned(base: pl.DataFrame, table: pl.DataFrame, keys: Sequence[str]) -> np.ndarray:
    keys = list(keys)
    joined = base.select(keys).join(table, on=keys, how="left", maintain_order="left")
    return column(joined, "value")


def to_series(name: str, values) -> pl.Series:
    if isinstance(values, pl.Series):
        return values.alias(name)
    arr = np.asarray(values, dtype=np.float64)
    return pl.Series(name, arr, dtype=pl.Float64).fill_nan(None)


def finish(
    base: pl.DataFrame,
    keys: Sequence[str],
    specs: Sequence[FeatureSpec],
    rules: Mapping[tuple[str, str], Rule],
) -> pl.DataFrame:
    cols = []
    for spec in specs:
        rule = rules.get((spec.signal, spec.agg))
        if rule is None:
            raise ValueError(
                f"no rule for feature {spec.name} ({spec.signal}, {spec.agg})"
            )
        cols.append(to_series(spec.name, rule(base, spec)))
    return base.select(keys).with_columns(cols)
