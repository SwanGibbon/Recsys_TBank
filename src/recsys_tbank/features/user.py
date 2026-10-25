import numpy as np
import polars as pl

from recsys_tbank.features.prep import (
    WINDOWS,
    Context,
    aligned,
    column,
    decay,
    finish,
    in_window,
    ratio,
    segments,
    ts_counts,
)
from recsys_tbank.features.spec import FeatureSpec, register_group, window_suffix
from recsys_tbank.split import DAY

GROUP = "U"
KEYS = ("uid",)


def _spec(name, signal, window, agg, *params) -> FeatureSpec:
    return FeatureSpec(
        name, GROUP, "user", signal, window, agg, tuple(map(float, params))
    )


SPECS = (
    *(_spec(f"u_lis_{window_suffix(w)}", "L", w, "count") for w in WINDOWS),
    *(_spec(f"u_pos_rate_{window_suffix(w)}", "P", w, "rate") for w in (7, 30, None)),
    _spec("u_org_share_30", "org", 30, "share"),
    _spec("u_org_share_all", "org", None, "share"),
    _spec("u_orgpos_rate_30", "PO", 30, "rate"),
    _spec("u_ratio_mean_30", "ratio", 30, "mean"),
    _spec("u_ratio_mean_all", "ratio", None, "mean"),
    _spec("u_nuniq_7", "item", 7, "nunique"),
    _spec("u_nuniq_30", "item", 30, "nunique"),
    _spec("u_novel_share_30", "first_listen", 30, "share"),
    _spec("u_recency", "event", None, "recency_days"),
    _spec("u_tenure", "event", None, "age_days"),
    *(
        _spec(f"u_active_days_{window_suffix(w)}", "event", w, "active_days")
        for w in (7, 30, None)
    ),
    _spec("u_decay_hl7", "L", None, "decay", 7),
    _spec("u_decay_hl30", "L", None, "decay", 30),
    _spec("u_like_net", "like", None, "net_nonburst"),
    _spec("u_like_30", "like", 30, "count"),
    _spec("u_burst_like_share", "like", None, "burst_share"),
    _spec("u_dislike_all", "dislike", None, "count"),
    _spec("u_unlike_all", "unlike", None, "count"),
    _spec("u_pop_taste", "item_pop", 30, "mean_log"),
    _spec("u_tl_mean_30", "track_len", 30, "mean"),
    _spec("u_tl_std_30", "track_len", 30, "std"),
    _spec("u_cold", "L", None, "cold", 50, 30),
    _spec("u_trend_7_30", "L", 30, "trend", 7),
    _spec("u_first_in_first_days", "event", None, "first_in_first_days", 7),
)


def _base(ctx: Context) -> pl.DataFrame:
    T = ctx.T
    et = pl.col("event_type")
    event_aggs = [pl.col("ts").min().alias("first"), pl.col("ts").max().alias("last")]
    listen_aggs = []
    novel_aggs = []
    for w in WINDOWS:
        s = window_suffix(w)
        c = in_window(T, w)
        event_aggs += [
            pl.col("day").filter(c).n_unique().alias(f"active_{s}"),
            ((et == "like") & c).sum().alias(f"like_{s}"),
            ((et == "dislike") & c).sum().alias(f"dislike_{s}"),
            ((et == "unlike") & c).sum().alias(f"unlike_{s}"),
        ]
        listen_aggs += [
            c.sum().alias(f"lis_{s}"),
            (c & pl.col("pos")).sum().alias(f"pos_{s}"),
            (c & pl.col("org")).sum().alias(f"org_{s}"),
            (c & pl.col("orgpos")).sum().alias(f"orgpos_{s}"),
            pl.col("ratio").filter(c).sum().alias(f"ratio_{s}"),
            pl.col("item_id").filter(c).n_unique().alias(f"nuniq_{s}"),
        ]
        novel_aggs.append(c.sum().alias(f"novel_{s}"))
    users = ctx.events.group_by("uid").agg(event_aggs)
    listens = ctx.listens.group_by("uid").agg(listen_aggs)
    novel = (
        ctx.pairs_all()
        .select("uid", pl.col("first").alias("ts"))
        .group_by("uid")
        .agg(novel_aggs)
    )
    burst = (
        ctx.likes().group_by("uid").agg(pl.col("is_burst").sum().alias("like_burst"))
    )
    base = users
    for part in (listens, novel, burst):
        base = base.join(part, on="uid", how="left")
    counts = [c for c in base.columns if c not in ("uid", "first", "last")]
    return base.with_columns(pl.col(c).fill_null(0) for c in counts).sort("uid")


def _pop_taste(ctx: Context, window: int | None) -> pl.DataFrame:
    own = (
        ctx.listens.filter(in_window(ctx.T, window))
        .group_by(["uid", "item_id"])
        .agg(pl.len().alias("c"))
        .join(ctx.item_listens(window), on="item_id", how="left")
        .sort(["uid", "item_id"])
    )
    group, start = segments(own, ["uid"])
    c = column(own, "c")
    size = int(start.sum())
    total = np.bincount(group, weights=c * np.log(column(own, "n")), minlength=size)
    n = np.bincount(group, weights=c, minlength=size)
    return (
        own.filter(pl.Series(start))
        .select("uid")
        .with_columns(pl.Series("value", ratio(total, n), dtype=pl.Float64))
    )


def _track_len(ctx: Context, window: int | None, agg: str) -> pl.DataFrame:
    counts = (
        ctx.listens.filter(in_window(ctx.T, window) & pl.col("tl").is_not_null())
        .group_by(["uid", "tl"])
        .agg(pl.len().alias("c"))
        .sort(["uid", "tl"])
    )
    group, start = segments(counts, ["uid"])
    size = int(start.sum())
    c = column(counts, "c")
    tl = column(counts, "tl")
    n = np.bincount(group, weights=c, minlength=size)
    mean = ratio(np.bincount(group, weights=c * tl, minlength=size), n)
    if agg == "mean":
        value = mean
    else:
        dev = np.bincount(group, weights=c * (tl - mean[group]) ** 2, minlength=size)
        value = np.sqrt(ratio(dev, n - 1))
    return (
        counts.filter(pl.Series(start))
        .select("uid")
        .with_columns(pl.Series("value", value, dtype=pl.Float64))
    )


def build(ctx: Context, specs) -> pl.DataFrame:
    T = ctx.T
    base = _base(ctx)
    keys = list(KEYS)

    def col(name):
        return column(base, name)

    def w(spec):
        return window_suffix(spec.window)

    def decayed(b, s):
        counts = ctx.memo("user_ts_counts", lambda: ts_counts(ctx.listens, keys))
        values = aligned(b, decay(counts, keys, T, s.params[0], s.window), keys)
        return np.nan_to_num(values, nan=0.0)

    def cold(b, s):
        lis, days = s.params
        return ((col("lis_all") < lis) | (T - col("first") < days * DAY)).astype(float)

    def trend(b, s):
        short = int(s.params[0])
        return ratio(col(f"lis_{short}") * s.window, col(f"lis_{w(s)}") * short)

    rules = {
        ("L", "count"): lambda b, s: col(f"lis_{w(s)}"),
        ("P", "rate"): lambda b, s: ratio(col(f"pos_{w(s)}"), col(f"lis_{w(s)}")),
        ("org", "share"): lambda b, s: ratio(col(f"org_{w(s)}"), col(f"lis_{w(s)}")),
        ("PO", "rate"): lambda b, s: ratio(col(f"orgpos_{w(s)}"), col(f"org_{w(s)}")),
        ("ratio", "mean"): lambda b, s: ratio(col(f"ratio_{w(s)}"), col(f"lis_{w(s)}")),
        ("item", "nunique"): lambda b, s: col(f"nuniq_{w(s)}"),
        ("first_listen", "share"): lambda b, s: ratio(
            col(f"novel_{w(s)}"), col(f"lis_{w(s)}")
        ),
        ("event", "recency_days"): lambda b, s: (T - col("last")) / DAY,
        ("event", "age_days"): lambda b, s: (T - col("first")) / DAY,
        ("event", "active_days"): lambda b, s: col(f"active_{w(s)}"),
        ("L", "decay"): decayed,
        ("like", "net_nonburst"): lambda b, s: (
            col("like_all") - col("like_burst") - col("unlike_all")
        ),
        ("like", "count"): lambda b, s: col(f"like_{w(s)}"),
        ("like", "burst_share"): lambda b, s: ratio(col("like_burst"), col("like_all")),
        ("dislike", "count"): lambda b, s: col(f"dislike_{w(s)}"),
        ("unlike", "count"): lambda b, s: col(f"unlike_{w(s)}"),
        ("item_pop", "mean_log"): lambda b, s: aligned(
            b, _pop_taste(ctx, s.window), keys
        ),
        ("track_len", "mean"): lambda b, s: aligned(
            b, _track_len(ctx, s.window, "mean"), keys
        ),
        ("track_len", "std"): lambda b, s: aligned(
            b, _track_len(ctx, s.window, "std"), keys
        ),
        ("L", "cold"): cold,
        ("L", "trend"): trend,
        ("event", "first_in_first_days"): lambda b, s: (
            col("first") < s.params[0] * DAY
        ).astype(float),
    }
    return finish(base, keys, specs, rules)


register_group(GROUP, KEYS, 10, build, SPECS)
