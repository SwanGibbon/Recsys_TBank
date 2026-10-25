import numpy as np
import polars as pl

from recsys_tbank.features.prep import (
    SMOOTH_M,
    WINDOWS,
    Context,
    aligned,
    column,
    decay,
    finish,
    global_rate,
    in_window,
    ratio,
    smoothed,
    ts_counts,
)
from recsys_tbank.features.spec import FeatureSpec, register_group, window_suffix
from recsys_tbank.split import DAY

GROUP = "I"
KEYS = ("item_id",)
M = SMOOTH_M


def _spec(name, signal, window, agg, *params) -> FeatureSpec:
    return FeatureSpec(
        name, GROUP, "item", signal, window, agg, tuple(map(float, params))
    )


SPECS = (
    *(_spec(f"i_lis_{window_suffix(w)}", "L", w, "count") for w in WINDOWS),
    *(_spec(f"i_users_{window_suffix(w)}", "L", w, "users") for w in (7, 30, None)),
    _spec("i_repeat_ratio", "L", None, "repeat"),
    _spec("i_pos_rate_s_30", "P", 30, "rate_s", M),
    _spec("i_pos_rate_s_all", "P", None, "rate_s", M),
    _spec("i_skip10_rate_s", "skip", None, "rate_s", M),
    _spec("i_org_share", "org", None, "share"),
    _spec("i_nonorg_conv_s", "nonorg_P", None, "rate_s", M),
    _spec("i_like_rate_s", "like", None, "rate_s", M),
    _spec("i_dislike_rate_s", "dislike", None, "rate_s", M),
    _spec("i_unlike_rate_s", "unlike", None, "rate_s", M),
    _spec("i_age", "event", None, "age_days"),
    _spec("i_recency", "event", None, "recency_days"),
    _spec("i_trend_7_30", "L", 30, "trend", 7),
    _spec("i_decay_hl7", "L", None, "decay", 7),
    _spec("i_track_len", "track_len", None, "last"),
    _spec("i_pop_rank_7d", "L", 7, "pop_rank"),
    _spec("i_cold", "L", None, "cold", 1),
)


def _base(ctx: Context) -> pl.DataFrame:
    T = ctx.T
    et = pl.col("event_type")
    uid = pl.col("uid")
    event_aggs = [
        pl.col("ts").min().alias("first"),
        pl.col("ts").max().alias("last"),
        *(
            uid.filter(et == k).n_unique().alias(f"{k}_users")
            for k in ("like", "dislike", "unlike")
        ),
    ]
    listen_aggs = []
    for w in WINDOWS:
        s = window_suffix(w)
        c = in_window(T, w)
        listen_aggs += [
            c.sum().alias(f"lis_{s}"),
            uid.filter(c).n_unique().alias(f"users_{s}"),
            (c & pl.col("pos")).sum().alias(f"pos_{s}"),
            (c & pl.col("skip")).sum().alias(f"skip_{s}"),
            (c & pl.col("org")).sum().alias(f"org_{s}"),
            (c & ~pl.col("org")).sum().alias(f"nonorg_{s}"),
            (c & pl.col("pos") & ~pl.col("org")).sum().alias(f"nonorg_pos_{s}"),
        ]
    items = ctx.events.group_by("item_id").agg(event_aggs)
    listens = ctx.listens.group_by("item_id").agg(listen_aggs)
    track = (
        ctx.listens.filter(pl.col("tl").is_not_null())
        .group_by("item_id")
        .agg(pl.col("tl").sort_by(["ts", "tl"]).last().alias("track_len"))
    )
    base = items.join(listens, on="item_id", how="left")
    counts = [c for c in base.columns if c not in ("item_id", "first", "last")]
    base = base.with_columns(pl.col(c).fill_null(0) for c in counts)
    return base.join(track, on="item_id", how="left").sort("item_id")


def build(ctx: Context, specs) -> pl.DataFrame:
    T = ctx.T
    base = _base(ctx)
    keys = list(KEYS)

    def col(name):
        return column(base, name)

    def w(spec):
        return window_suffix(spec.window)

    def rate_s(k, n, spec):
        return smoothed(k, n, global_rate(k, n), spec.params[0])

    def decayed(b, s):
        counts = ctx.memo("item_ts_counts", lambda: ts_counts(ctx.listens, keys))
        values = aligned(b, decay(counts, keys, T, s.params[0], s.window), keys)
        return np.nan_to_num(values, nan=0.0)

    def trend(b, s):
        short = int(s.params[0])
        return ratio(col(f"lis_{short}") * s.window, col(f"lis_{w(s)}") * short)

    def pop_rank(b, s):
        return b[f"lis_{w(s)}"].rank("min", descending=True).cast(pl.Float64)

    rules = {
        ("L", "count"): lambda b, s: col(f"lis_{w(s)}"),
        ("L", "users"): lambda b, s: col(f"users_{w(s)}"),
        ("L", "repeat"): lambda b, s: ratio(col(f"lis_{w(s)}"), col(f"users_{w(s)}")),
        ("P", "rate_s"): lambda b, s: rate_s(col(f"pos_{w(s)}"), col(f"lis_{w(s)}"), s),
        ("skip", "rate_s"): lambda b, s: rate_s(
            col(f"skip_{w(s)}"), col(f"lis_{w(s)}"), s
        ),
        ("org", "share"): lambda b, s: ratio(col(f"org_{w(s)}"), col(f"lis_{w(s)}")),
        ("nonorg_P", "rate_s"): lambda b, s: rate_s(
            col(f"nonorg_pos_{w(s)}"), col(f"nonorg_{w(s)}"), s
        ),
        ("like", "rate_s"): lambda b, s: rate_s(
            col("like_users"), col(f"users_{w(s)}"), s
        ),
        ("dislike", "rate_s"): lambda b, s: rate_s(
            col("dislike_users"), col(f"users_{w(s)}"), s
        ),
        ("unlike", "rate_s"): lambda b, s: rate_s(
            col("unlike_users"), col(f"users_{w(s)}"), s
        ),
        ("event", "age_days"): lambda b, s: (T - col("first")) / DAY,
        ("event", "recency_days"): lambda b, s: (T - col("last")) / DAY,
        ("L", "trend"): trend,
        ("L", "decay"): decayed,
        ("track_len", "last"): lambda b, s: col("track_len"),
        ("L", "pop_rank"): pop_rank,
        ("L", "cold"): lambda b, s: (col(f"users_{w(s)}") <= s.params[0]).astype(float),
    }
    return finish(base, keys, specs, rules)


register_group(GROUP, KEYS, 20, build, SPECS)
