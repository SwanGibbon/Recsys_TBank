import operator
from pathlib import Path
from typing import NamedTuple

import polars as pl

DAY = 86400
WINDOW_DAYS = 7
WINDOW_SEC = WINDOW_DAYS * DAY
BUCKET_SEC = 600
BURST_MIN = 20
THRESHOLD = 50
TRAIN_DAYS = (266, 273, 280)
VAL_DAY = 287
TEST_DAY = 294
TRAIN_TS = tuple(d * DAY for d in TRAIN_DAYS)
VAL_TS = VAL_DAY * DAY
TEST_TS = TEST_DAY * DAY
DATA_END = 26_000_000
DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "dev" / "events_1000u.parquet"
)
SORT_COLUMNS = (
    "timestamp",
    "uid",
    "item_id",
    "event_type",
    "is_organic",
    "played_ratio_pct",
    "track_length_seconds",
)

Frame = pl.DataFrame | pl.LazyFrame

_TS = pl.col("timestamp")
_IS_LISTEN = pl.col("event_type") == "listen"
_IS_POSITIVE = _IS_LISTEN & (pl.col("played_ratio_pct") >= THRESHOLD)
_PAIR = ["uid", "item_id"]


class Window(NamedTuple):
    name: str
    T: int
    T_end: int


class LeakageError(AssertionError):
    pass


def windows() -> tuple[Window, ...]:
    starts = [(f"train_{d}", d * DAY) for d in TRAIN_DAYS] + [("val", VAL_TS)]
    out = [Window(name, t, t + WINDOW_SEC) for name, t in starts]
    out.append(Window("test", TEST_TS, DATA_END + 1))
    return tuple(out)


def get_window(name: str) -> Window:
    for w in windows():
        if w.name == name:
            return w
    raise KeyError(name)


def check_test_access(upper: int, allow_test: bool = False) -> None:
    if operator.index(upper) > TEST_TS and not allow_test:
        raise PermissionError(
            f"reading events with timestamp >= {TEST_TS} (test window) requires allow_test=True"
        )


def _require_frame(events) -> None:
    if not isinstance(events, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            f"expected polars DataFrame or LazyFrame, got {type(events).__name__}"
        )


def _cut(T) -> int:
    T = operator.index(T)
    if T % BUCKET_SEC:
        raise ValueError(f"cut {T} is not a multiple of BUCKET_SEC={BUCKET_SEC}")
    return T


def _lazy(events: Frame) -> pl.LazyFrame:
    _require_frame(events)
    return events.lazy()


def load_events(path: str | Path = DATA_PATH) -> pl.LazyFrame:
    return pl.scan_parquet(path).sort(list(SORT_COLUMNS), maintain_order=True)


def history(events: Frame, T: int, allow_test: bool = False) -> Frame:
    _require_frame(events)
    T = _cut(T)
    check_test_access(T, allow_test)
    return events.filter(_TS < T)


def label_events(events: Frame, T: int, T_end: int, allow_test: bool = False) -> Frame:
    _require_frame(events)
    T = _cut(T)
    T_end = operator.index(T_end)
    if T_end <= T:
        raise ValueError(f"empty label window [{T}, {T_end})")
    check_test_access(T_end, allow_test)
    return events.filter((_TS >= T) & (_TS < T_end))


def assert_no_future(frame: Frame, T: int) -> None:
    latest = _lazy(frame).select(_TS.max()).collect().item()
    if latest is not None and latest >= T:
        raise LeakageError(f"max timestamp {latest} >= cut {T}")


def positive_pairs(
    events: Frame, T: int, T_end: int, allow_test: bool = False
) -> pl.DataFrame:
    return (
        _lazy(label_events(events, T, T_end, allow_test))
        .filter(_IS_POSITIVE)
        .group_by(_PAIR)
        .agg((pl.col("is_organic") == 1).any().cast(pl.UInt8).alias("y_org"))
        .with_columns(pl.lit(1, dtype=pl.UInt8).alias("y_all"))
        .sort(_PAIR)
        .collect()
    )


def novel_flag(history_events: Frame, pairs: Frame) -> pl.DataFrame:
    seen = (
        _lazy(history_events)
        .filter(_IS_LISTEN)
        .select(_PAIR)
        .unique()
        .with_columns(pl.lit(0, dtype=pl.UInt8).alias("novel"))
    )
    return (
        _lazy(pairs)
        .join(seen, on=_PAIR, how="left", maintain_order="left")
        .with_columns(pl.col("novel").fill_null(1))
        .collect()
    )


def net_like_state(events_hist: Frame) -> pl.DataFrame:
    return (
        _lazy(events_hist)
        .filter(pl.col("event_type").is_in(["like", "unlike"]))
        .with_columns((pl.col("event_type") == "unlike").cast(pl.UInt8).alias("_tie"))
        .group_by(_PAIR)
        .agg(
            (pl.col("event_type").sort_by(["timestamp", "_tie"]).last() == "like")
            .cast(pl.UInt8)
            .alias("liked")
        )
        .sort(_PAIR)
        .collect()
    )


def burst_flags(events_hist: Frame) -> pl.DataFrame:
    return (
        _lazy(events_hist)
        .filter(pl.col("event_type") == "like")
        .group_by(pl.col("uid"), (_TS // BUCKET_SEC).alias("bucket"))
        .agg(pl.len().alias("n"))
        .select(
            "uid",
            "bucket",
            (pl.col("n") >= BURST_MIN).cast(pl.UInt8).alias("is_burst"),
        )
        .sort("uid", "bucket")
        .collect()
    )


def window_summary(events: Frame, allow_test: bool = False) -> pl.DataFrame:
    _require_frame(events)
    df = events.collect() if isinstance(events, pl.LazyFrame) else events
    rows = []
    for w in windows():
        if w.name == "test" and not allow_test:
            continue
        hist = history(df, w.T)
        lab = label_events(df, w.T, w.T_end, allow_test)
        pairs = positive_pairs(df, w.T, w.T_end, allow_test)
        hist_users = hist.select("uid").unique()
        rows.append(
            {
                "name": w.name,
                "T": w.T,
                "T_end": w.T_end,
                "history_rows": hist.height,
                "history_users": hist_users.height,
                "label_listen_rows": lab.filter(_IS_LISTEN).height,
                "pos_pairs_org": int(pairs["y_org"].sum()),
                "pos_pairs_all": pairs.height,
                "users_pos_org": pairs.filter(pl.col("y_org") == 1)["uid"].n_unique(),
                "users_pos_all": pairs["uid"].n_unique(),
                "new_users": lab.select("uid")
                .unique()
                .join(hist_users, on="uid", how="anti")
                .height,
            }
        )
    schema = {k: pl.String if k == "name" else pl.Int64 for k in rows[0]}
    return pl.DataFrame(rows, schema=schema)
