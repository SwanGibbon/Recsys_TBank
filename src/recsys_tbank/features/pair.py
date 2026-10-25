import numpy as np
import polars as pl

from recsys_tbank.features.prep import (
    PAIR,
    WINDOWS,
    Context,
    aligned,
    column,
    decay,
    finish,
    in_window,
    ratio,
    ts_counts,
)
from recsys_tbank.features.spec import FeatureSpec, register_group, window_suffix
from recsys_tbank.split import DAY

GROUP = "D"
KEYS = tuple(PAIR)


def _spec(name, signal, window, agg, *params) -> FeatureSpec:
    return FeatureSpec(
        name, GROUP, "pair", signal, window, agg, tuple(map(float, params))
    )


SPECS = (
    *(_spec(f"ui_lis_{window_suffix(w)}", "L", w, "count") for w in WINDOWS),
    *(_spec(f"ui_pos_{window_suffix(w)}", "P", w, "count") for w in (7, 30, None)),
    *(_spec(f"ui_orgpos_{window_suffix(w)}", "PO", w, "count") for w in (30, None)),
    _spec("ui_pos_rate", "P", None, "rate"),
    _spec("ui_skip_all", "skip", None, "rate"),
    _spec("ui_org_share", "org", None, "share"),
    _spec("ui_ratio_last", "ratio", None, "last"),
    _spec("ui_ratio_max", "ratio", None, "max"),
    _spec("ui_ratio_mean", "ratio", None, "mean"),
    _spec("ui_rec_last", "L", None, "recency_days"),
    _spec("ui_rec_last_pos", "P", None, "recency_days"),
    _spec("ui_age_first", "L", None, "age_days"),
    _spec("ui_active_days", "L", None, "active_days"),
    _spec("ui_decay_org_hl14", "PO", 90, "decay", 14),
    _spec("ui_decay_all_hl7", "L", None, "decay", 7),
    _spec("ui_like_state", "like", None, "state"),
    _spec("ui_like_burst", "like", None, "burst_last"),
    _spec("ui_dislike_state", "dislike", None, "state"),
    _spec("ui_share_u_30", "L", 30, "share_user"),
    _spec("ui_rank_user_all", "L", None, "rank_user"),
)

_SIGNAL_FILTER = {"L": None, "P": "pos", "PO": "orgpos"}


def _base(ctx: Context, listens: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    T = ctx.T
    aggs = [
        pl.col("ts").max().alias("last"),
        pl.col("ts").filter(pl.col("pos")).max().alias("last_P"),
        pl.col("ts").min().alias("first"),
        pl.col("ratio").max().alias("ratio_max"),
        pl.col("ratio").sort_by(["ts", "ratio"]).last().alias("ratio_last"),
    ]
    for w in WINDOWS:
        s = window_suffix(w)
        c = in_window(T, w)
        aggs += [
            c.sum().alias(f"lis_{s}"),
            (c & pl.col("pos")).sum().alias(f"pos_{s}"),
            (c & pl.col("orgpos")).sum().alias(f"orgpos_{s}"),
            (c & pl.col("skip")).sum().alias(f"skip_{s}"),
            (c & pl.col("org")).sum().alias(f"org_{s}"),
            pl.col("ratio").filter(c).sum().alias(f"ratio_{s}"),
            pl.col("day").filter(c).n_unique().alias(f"active_{s}"),
        ]
    stats = listens.group_by(PAIR).agg(aggs)
    return pairs.join(stats, on=PAIR, how="left").sort(PAIR)


def _last_state(ctx: Context, on: str, off: str) -> pl.DataFrame:
    return (
        ctx.events.filter(pl.col("event_type").is_in([on, off]))
        .with_columns((pl.col("event_type") == off).cast(pl.UInt8).alias("_tie"))
        .group_by(PAIR)
        .agg(
            (pl.col("event_type").sort_by(["ts", "_tie"]).last() == on)
            .cast(pl.Int64)
            .alias("value")
        )
    )


def build(ctx: Context, specs) -> pl.DataFrame:
    T = ctx.T
    keys = list(KEYS)
    pairs = ctx.keys.unique()
    listens = ctx.listens.join(pairs, on=PAIR, how="semi")
    base = _base(ctx, listens, pairs)

    def col(name):
        return column(base, name)

    def w(spec):
        return window_suffix(spec.window)

    def decayed(b, s):
        flag = _SIGNAL_FILTER[s.signal]

        def make():
            sub = listens if flag is None else listens.filter(pl.col(flag))
            return ts_counts(sub, keys)

        counts = ctx.memo(f"pair_ts_counts_{s.signal}", make)
        table = decay(counts, keys, T, s.params[0], s.window)
        return aligned(b, table, keys)

    def fill_listened(values):
        out = values.copy()
        out[(col("lis_all") > 0) & np.isnan(out)] = 0.0
        return out

    def like_state(b, s):
        state = ctx.like_state().rename({"liked": "value"})
        return aligned(b, state, keys)

    def like_burst(b, s):
        last = (
            ctx.likes()
            .group_by(PAIR)
            .agg(pl.col("is_burst").sort_by(["ts", "is_burst"]).last().alias("value"))
        )
        return aligned(b, last, keys)

    def share_user(b, s):
        users = ctx.user_listens(s.window).rename({"n": "value"})
        return ratio(col(f"lis_{w(s)}"), aligned(b, users, ["uid"]))

    def rank_user(b, s):
        ranked = (
            ctx.pairs_all()
            .with_columns(
                pl.col("n").rank("min", descending=True).over("uid").alias("value")
            )
            .select(*PAIR, "value")
        )
        return aligned(b, ranked, keys)

    rules = {
        ("L", "count"): lambda b, s: col(f"lis_{w(s)}"),
        ("P", "count"): lambda b, s: col(f"pos_{w(s)}"),
        ("PO", "count"): lambda b, s: col(f"orgpos_{w(s)}"),
        ("P", "rate"): lambda b, s: ratio(col(f"pos_{w(s)}"), col(f"lis_{w(s)}")),
        ("skip", "rate"): lambda b, s: ratio(col(f"skip_{w(s)}"), col(f"lis_{w(s)}")),
        ("org", "share"): lambda b, s: ratio(col(f"org_{w(s)}"), col(f"lis_{w(s)}")),
        ("ratio", "last"): lambda b, s: col("ratio_last"),
        ("ratio", "max"): lambda b, s: col("ratio_max"),
        ("ratio", "mean"): lambda b, s: ratio(col(f"ratio_{w(s)}"), col(f"lis_{w(s)}")),
        ("L", "recency_days"): lambda b, s: (T - col("last")) / DAY,
        ("P", "recency_days"): lambda b, s: (T - col("last_P")) / DAY,
        ("L", "age_days"): lambda b, s: (T - col("first")) / DAY,
        ("L", "active_days"): lambda b, s: col(f"active_{w(s)}"),
        ("PO", "decay"): lambda b, s: fill_listened(decayed(b, s)),
        ("L", "decay"): lambda b, s: fill_listened(decayed(b, s)),
        ("like", "state"): like_state,
        ("like", "burst_last"): like_burst,
        ("dislike", "state"): lambda b, s: aligned(
            b, _last_state(ctx, "dislike", "undislike"), keys
        ),
        ("L", "share_user"): share_user,
        ("L", "rank_user"): rank_user,
    }
    return finish(base, keys, specs, rules)


register_group(GROUP, KEYS, 30, build, SPECS)
