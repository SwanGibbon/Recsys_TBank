import numpy as np
import polars as pl

from recsys_tbank.candidates import SOURCES
from recsys_tbank.features.prep import PAIR, Context, column, finish, ratio
from recsys_tbank.features.spec import FeatureSpec, register_group

SCORED = ("als", "i2i")
SOURCE_COLUMNS = (
    *(f"in_{s}" for s in SOURCES),
    *(f"rank_{s}" for s in SOURCES),
    *(f"score_{s}" for s in SCORED),
    "n_sources",
)
_AGG = {"in": "flag", "rank": "rank", "score": "score", "n": "count"}

SPECS = tuple(
    FeatureSpec(c, "C", "pair", c.partition("_")[2], None, _AGG[c.partition("_")[0]])
    for c in SOURCE_COLUMNS
)
CONTEXT_SPECS = (
    FeatureSpec("n_cands_user", "Z", "user", "cands", None, "count"),
    FeatureSpec("share_user_src", "Z", "user", "user", None, "share"),
)


def _pool(ctx: Context) -> pl.DataFrame:
    def make() -> pl.DataFrame:
        present = [c for c in SOURCE_COLUMNS if c in ctx.cands.columns]
        return (
            pl.concat([ctx.keys, ctx.cands.select(present)], how="horizontal")
            .unique(PAIR, keep="first", maintain_order=True)
            .sort(PAIR)
        )

    return ctx.memo("pool", make)


def _copy(b: pl.DataFrame, spec: FeatureSpec) -> np.ndarray:
    if spec.name not in b.columns:
        return np.full(b.height, np.nan)
    return column(b, spec.name)


def build_source(ctx: Context, specs) -> pl.DataFrame:
    rules = {(s.signal, s.agg): _copy for s in SPECS}
    return finish(_pool(ctx), PAIR, specs, rules)


def build_context(ctx: Context, specs) -> pl.DataFrame:
    pool = _pool(ctx)
    flag = "in_user" if "in_user" in pool.columns else None
    base = (
        pool.group_by("uid")
        .agg(
            pl.len().alias("n"),
            (pl.col(flag).cast(pl.Int64).sum() if flag else pl.lit(None)).alias("k"),
        )
        .join(ctx.events.select("uid").unique(), on="uid", how="semi")
        .sort("uid")
    )
    rules = {
        ("cands", "count"): lambda b, s: column(b, "n"),
        ("user", "share"): lambda b, s: ratio(column(b, "k"), column(b, "n")),
    }
    return finish(base, ["uid"], specs, rules)


register_group("C", PAIR, 60, build_source, SPECS)
register_group("Z", ("uid",), 70, build_context, CONTEXT_SPECS)
