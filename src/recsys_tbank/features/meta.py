import hashlib
import os
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from recsys_tbank.features.prep import (
    PAIR,
    SMOOTH_M,
    Context,
    column,
    finish,
    global_rate,
    in_window,
    ratio,
    smoothed,
)
from recsys_tbank.features.spec import FeatureSpec, register_group, window_suffix
from recsys_tbank.split import DAY

GROUP = "M"
KEYS = tuple(PAIR)
OWNERS = ("artist", "album")
OWNER_WINDOWS = (7, 30, None)
META_DIR = Path(__file__).resolve().parents[3] / "data" / "yambda" / "meta"
MAPPING_PATHS = {o: META_DIR / f"{o}_item_mapping.parquet" for o in OWNERS}
ALLOW_NO_MAPPING = "RECSYS_ALLOW_NO_MAPPING"
_METRIC = {"L": "lis", "P": "pos", "PO": "orgpos"}


class ItemMapping(NamedTuple):
    items: pl.DataFrame
    sha256: str


def _spec(name, entity, signal, window, agg, *params) -> FeatureSpec:
    return FeatureSpec(
        name, GROUP, entity, signal, window, agg, tuple(map(float, params))
    )


SPECS = (
    *(
        _spec(f"ua_lis_{window_suffix(w)}", "user_artist", "artist.L", w, "count")
        for w in (7, 30, None)
    ),
    *(
        _spec(f"ua_pos_{window_suffix(w)}", "user_artist", "artist.P", w, "count")
        for w in (30, None)
    ),
    _spec("ua_orgpos_30", "user_artist", "artist.PO", 30, "count"),
    _spec("ua_share_30", "user_artist", "artist.L", 30, "share_user"),
    _spec("ua_like_net", "user_artist", "artist.like", None, "net_nonburst"),
    _spec("ua_rec_last", "user_artist", "artist.L", None, "recency_days"),
    _spec("ua_n_items", "user_artist", "artist.item", None, "nunique"),
    _spec("ub_lis_30", "user_album", "album.L", 30, "count"),
    _spec("ub_pos_all", "user_album", "album.P", None, "count"),
    _spec("ub_rec_last", "user_album", "album.L", None, "recency_days"),
    _spec("ub_like_net", "user_album", "album.like", None, "net_nonburst"),
    *(
        _spec(f"ia_pop_{window_suffix(w)}", "artist", "artist.L", w, "users")
        for w in (30, None)
    ),
    _spec("ia_pos_rate_s", "artist", "artist.P", None, "rate_s", SMOOTH_M),
    _spec("ia_n_items", "artist", "artist.item", None, "nunique_global"),
    _spec("ui_known_artist_new_item", "pair", "artist.L", None, "known_new_item"),
    _spec("n_artists", "item", "artist", None, "count"),
    _spec("n_albums", "item", "album", None, "count"),
    _spec("m_missing", "item", "mapping", None, "missing"),
)


def _file_sha256(paths) -> str:
    digest = hashlib.sha256()
    for name in sorted(paths):
        path = Path(paths[name])
        digest.update(name.encode())
        if path.exists():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def _per_item(path: Path, owner: str) -> pl.DataFrame:
    key = f"{owner}_id"
    count = f"n_{owner}s"
    schema = {"item_id": pl.UInt32, key: pl.UInt32, count: pl.UInt32}
    if not path.exists():
        return pl.DataFrame(schema=schema)
    return (
        pl.read_parquet(path, columns=[key, "item_id"])
        .select(pl.col("item_id").cast(pl.UInt32), pl.col(key).cast(pl.UInt32))
        .drop_nulls()
        .group_by("item_id")
        .agg(
            pl.col(key).min(),
            pl.col(key).n_unique().cast(pl.UInt32).alias(count),
        )
        .select(list(schema))
    )


def read_mapping(paths=MAPPING_PATHS) -> ItemMapping:
    absent = sorted(str(Path(paths[o])) for o in OWNERS if not Path(paths[o]).exists())
    if absent and os.environ.get(ALLOW_NO_MAPPING) != "1":
        raise FileNotFoundError(
            f"artist/album mapping missing: {absent}; download the Yambda "
            f"*_item_mapping.parquet files to {META_DIR} or set {ALLOW_NO_MAPPING}=1 "
            "to build M features with m_missing = 1 for every item"
        )
    artists, albums = (_per_item(Path(paths[o]), o) for o in OWNERS)
    items = (
        artists.join(albums, on="item_id", how="full", coalesce=True)
        .with_columns(pl.col("n_artists", "n_albums").fill_null(0))
        .sort("item_id")
    )
    return ItemMapping(items, _file_sha256(paths))


@cache
def load_mapping() -> ItemMapping:
    return read_mapping()


def mapping_sha256() -> str:
    return load_mapping().sha256


def _items(ctx: Context) -> pl.DataFrame:
    def make() -> pl.DataFrame:
        seen = ctx.events.select("item_id").unique()
        return seen.join(load_mapping().items, on="item_id", how="left").with_columns(
            pl.col("n_artists", "n_albums").fill_null(0)
        )

    return ctx.memo("meta_items", make)


def _listens(ctx: Context) -> pl.DataFrame:
    return ctx.memo(
        "meta_listens",
        lambda: ctx.listens.join(
            _items(ctx).select("item_id", *(f"{o}_id" for o in OWNERS)),
            on="item_id",
            how="left",
        ),
    )


def _liked(ctx: Context) -> pl.DataFrame:
    def make() -> pl.DataFrame:
        last = (
            ctx.likes()
            .group_by(PAIR)
            .agg(pl.col("is_burst").sort_by(["ts", "is_burst"]).last().alias("burst"))
        )
        return (
            ctx.like_state()
            .filter(pl.col("liked") == 1)
            .join(last, on=PAIR, how="left")
            .filter(pl.col("burst").fill_null(0) == 0)
            .select(PAIR)
        )

    return ctx.memo("meta_liked", make)


def _user_owner(ctx: Context, owner: str) -> pl.DataFrame:
    key = f"{owner}_id"
    aggs = [
        pl.col("ts").max().alias("last"),
        pl.col("item_id").n_unique().alias("n_items"),
    ]
    for w in OWNER_WINDOWS:
        s = window_suffix(w)
        c = in_window(ctx.T, w)
        aggs += [
            c.sum().alias(f"lis_{s}"),
            (c & pl.col("pos")).sum().alias(f"pos_{s}"),
            (c & pl.col("orgpos")).sum().alias(f"orgpos_{s}"),
        ]
    stats = (
        _listens(ctx).filter(pl.col(key).is_not_null()).group_by(["uid", key]).agg(aggs)
    )
    liked = (
        _liked(ctx)
        .join(_items(ctx).select("item_id", key), on="item_id", how="inner")
        .filter(pl.col(key).is_not_null())
        .group_by(["uid", key])
        .agg(pl.len().alias("liked"))
    )
    return stats.join(liked, on=["uid", key], how="full", coalesce=True)


def _owner(ctx: Context, owner: str) -> pl.DataFrame:
    key = f"{owner}_id"

    def make() -> pl.DataFrame:
        aggs = [pl.col("item_id").n_unique().alias("g_items")]
        for w in OWNER_WINDOWS:
            s = window_suffix(w)
            c = in_window(ctx.T, w)
            aggs += [
                c.sum().alias(f"g_lis_{s}"),
                (c & pl.col("pos")).sum().alias(f"g_pos_{s}"),
                pl.col("uid").filter(c).n_unique().alias(f"g_users_{s}"),
            ]
        return (
            _listens(ctx)
            .filter(pl.col(key).is_not_null())
            .group_by(key)
            .agg(aggs)
            .sort(key)
        )

    return ctx.memo(f"meta_owner_{owner}", make)


def _prefixed(frame: pl.DataFrame, owner: str, keep) -> pl.DataFrame:
    return frame.rename({c: f"{owner}_{c}" for c in frame.columns if c not in keep})


def _base(ctx: Context) -> pl.DataFrame:
    base = ctx.keys.unique().join(_items(ctx), on="item_id", how="left")
    for owner in OWNERS:
        key = f"{owner}_id"
        user = _prefixed(_user_owner(ctx, owner), owner, ("uid", key))
        glob = _prefixed(_owner(ctx, owner), owner, (key,))
        base = base.join(user, on=["uid", key], how="left").join(
            glob, on=key, how="left"
        )
    pair = ctx.pairs_all().select(*PAIR, pl.col("n").alias("pair_lis"))
    return base.join(pair, on=PAIR, how="left").sort(PAIR)


def build(ctx: Context, specs) -> pl.DataFrame:
    T = ctx.T
    keys = list(KEYS)
    base = _base(ctx)

    def col(name):
        return column(base, name)

    has = {o: ~np.isnan(col(f"{o}_id")) for o in OWNERS}

    def parts(spec):
        owner, _, signal = spec.signal.partition(".")
        return owner, signal, window_suffix(spec.window)

    def owned(owner, values):
        return np.where(has[owner], np.nan_to_num(values, nan=0.0), np.nan)

    def count(b, s):
        o, sig, w = parts(s)
        return owned(o, col(f"{o}_{_METRIC[sig]}_{w}"))

    def share_user(b, s):
        o, _, w = parts(s)
        users = ctx.user_listens(s.window).rename({"n": "value"})
        den = column(b.select("uid").join(users, on="uid", how="left"), "value")
        return np.where(has[o], ratio(np.nan_to_num(col(f"{o}_lis_{w}")), den), np.nan)

    def recency(b, s):
        o, _, _ = parts(s)
        return np.where(has[o], (T - col(f"{o}_last")) / DAY, np.nan)

    def rate_s(b, s):
        o, sig, w = parts(s)
        table = _owner(ctx, o)
        k = f"g_{_METRIC[sig]}_{w}"
        n = f"g_lis_{w}"
        prior = global_rate(column(table, k), column(table, n))
        k_b = np.nan_to_num(col(f"{o}_{k}"))
        n_b = np.nan_to_num(col(f"{o}_{n}"))
        return owned(o, smoothed(k_b, n_b, prior, s.params[0]))

    def known_new(b, s):
        o, _, _ = parts(s)
        heard = np.nan_to_num(col(f"{o}_lis_all")) > 0
        fresh = np.nan_to_num(col("pair_lis")) == 0
        return np.where(has[o], (heard & fresh).astype(float), np.nan)

    def missing(b, s):
        flag = (col("n_artists") == 0) | (col("n_albums") == 0)
        return np.where(np.isnan(col("n_artists")), np.nan, flag.astype(float))

    rules = {
        ("artist.L", "count"): count,
        ("artist.P", "count"): count,
        ("artist.PO", "count"): count,
        ("album.L", "count"): count,
        ("album.P", "count"): count,
        ("artist.L", "share_user"): share_user,
        ("artist.like", "net_nonburst"): lambda b, s: owned(
            "artist", col("artist_liked")
        ),
        ("album.like", "net_nonburst"): lambda b, s: owned("album", col("album_liked")),
        ("artist.L", "recency_days"): recency,
        ("album.L", "recency_days"): recency,
        ("artist.item", "nunique"): lambda b, s: owned("artist", col("artist_n_items")),
        ("artist.L", "users"): lambda b, s: owned(
            "artist", col(f"artist_g_users_{window_suffix(s.window)}")
        ),
        ("artist.P", "rate_s"): rate_s,
        ("artist.item", "nunique_global"): lambda b, s: owned(
            "artist", col("artist_g_items")
        ),
        ("artist.L", "known_new_item"): known_new,
        ("artist", "count"): lambda b, s: col("n_artists"),
        ("album", "count"): lambda b, s: col("n_albums"),
        ("mapping", "missing"): missing,
    }
    return finish(base, keys, specs, rules)


register_group(GROUP, KEYS, 40, build, SPECS)
