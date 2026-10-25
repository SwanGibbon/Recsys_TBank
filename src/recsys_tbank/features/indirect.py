import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import implicit
import numpy as np
import polars as pl
import scipy.sparse as sps

from recsys_tbank import candidates as cd
from recsys_tbank.features import item, user
from recsys_tbank.features.prep import PAIR, Context, column, finish, ratio, segments
from recsys_tbank.features.spec import FeatureSpec, register_group

GROUP = "X"
KEYS = tuple(PAIR)
ANCHORS = cd.DEFAULT_CONFIG.i2i_anchor_items
FACTORS_VERSION = "als-factors-1"
_FACTOR_DIR: ContextVar[Path | None] = ContextVar("factor_cache_dir", default=None)


def _spec(name, signal, window, agg, *params) -> FeatureSpec:
    return FeatureSpec(
        name, GROUP, "pair", signal, window, agg, tuple(map(float, params))
    )


SPECS = (
    _spec("cooc_sum", "cooc", None, "sum", ANCHORS),
    _spec("cooc_max", "cooc", None, "max", ANCHORS),
    _spec("cooc_max_last5", "cooc", None, "max", 5),
    _spec("als_dot", "als", None, "dot"),
    _spec("gap_pop", "item_pop", 30, "gap"),
    _spec("gap_tl_z", "track_len", 30, "z"),
    _spec("org_match", "org", 30, "product"),
)

_USER_INPUTS = ("u_pop_taste", "u_tl_mean_30", "u_tl_std_30", "u_org_share_30")
_ITEM_INPUTS = ("i_lis_30", "i_track_len", "i_org_share")


@contextmanager
def factor_cache(cache_dir):
    token = _FACTOR_DIR.set(None if cache_dir is None else Path(cache_dir))
    try:
        yield
    finally:
        _FACTOR_DIR.reset(token)


def factors_key(hist, T: int, cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG) -> str:
    from recsys_tbank.features.cache import hist_fingerprint

    payload = {
        "version": FACTORS_VERSION,
        "T": int(T),
        "hist": hist_fingerprint(hist),
        "cand_config": cfg.config_dict(),
        "candidates_sha256": hashlib.sha256(Path(cd.__file__).read_bytes()).hexdigest(),
        "implicit": implicit.__version__,
        "numpy": np.__version__,
        "polars": pl.__version__,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def factors_path(key: str, cache_dir) -> Path:
    return Path(cache_dir) / f"als-{key}.npz"


def load_factors(
    hist, T: int, cfg: cd.CandidateConfig = cd.DEFAULT_CONFIG, cache_dir=None
) -> cd.ALSFactors:
    if cache_dir is None:
        return cd.als_factors(hist, T, cfg)
    path = factors_path(factors_key(hist, T, cfg), cache_dir)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return cd.ALSFactors(*(data[f] for f in cd.ALSFactors._fields))
    factors = cd.als_factors(hist, T, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **factors._asdict())
    os.replace(tmp, path)
    return factors


def _factors(ctx: Context) -> cd.ALSFactors:
    cache_dir = _FACTOR_DIR.get()
    return ctx.memo(
        "als_factors", lambda: load_factors(ctx.hist, ctx.T, cache_dir=cache_dir)
    )


def _item_user(ctx: Context) -> tuple[np.ndarray, sps.csr_matrix]:
    pos = ctx.listens.filter(pl.col("pos")).select(PAIR).unique().sort(PAIR)
    items = pos["item_id"].unique().sort()
    users = pos["uid"].unique().sort()
    rows = np.searchsorted(items.to_numpy(), pos["item_id"].to_numpy())
    cols = np.searchsorted(users.to_numpy(), pos["uid"].to_numpy())
    binary = sps.csr_matrix(
        (np.ones(pos.height, dtype=np.float64), (rows, cols)),
        shape=(items.len(), users.len()),
    )
    norms = np.sqrt(np.asarray(binary.sum(axis=1)).ravel())
    inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    matrix = (sps.diags(inv) @ binary).tocsr()
    matrix.sort_indices()
    return items.to_numpy(), matrix


def _anchors(ctx: Context, depth: int) -> pl.DataFrame:
    return (
        ctx.listens.group_by(PAIR)
        .agg(pl.col("ts").max().alias("last"))
        .sort(["uid", "last", "item_id"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("uid").alias("rank"))
        .filter(pl.col("rank") <= depth)
        .sort(["uid", "rank"])
    )


def _row_index(items: np.ndarray, values: np.ndarray) -> np.ndarray:
    if items.size == 0:
        return np.full(values.size, -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(items, values), items.size - 1)
    return np.where(items[pos] == values, pos, -1)


def _cooc(ctx: Context, base: pl.DataFrame, depths: tuple[int, ...]) -> dict:
    depth = max(depths)
    items, matrix = _item_user(ctx)
    anchors = _anchors(ctx, depth)
    a_uid = anchors["uid"].to_numpy()
    a_item = anchors["item_id"].to_numpy()
    a_row = _row_index(items, a_item)
    c_uid = base["uid"].to_numpy()
    c_item = base["item_id"].to_numpy()
    c_row = _row_index(items, c_item)
    out = {
        k: (np.full(base.height, np.nan), np.full(base.height, np.nan)) for k in depths
    }
    _, start = segments(base, ["uid"])
    bounds = np.append(np.flatnonzero(start), base.height)
    for g in range(bounds.size - 1):
        lo, hi = bounds[g], bounds[g + 1]
        u = c_uid[lo]
        a_lo, a_hi = np.searchsorted(a_uid, u), np.searchsorted(a_uid, u, side="right")
        if a_lo == a_hi:
            continue
        ar, cr = a_row[a_lo:a_hi], c_row[lo:hi]
        sims = np.zeros((ar.size, cr.size))
        a_ok, c_ok = np.flatnonzero(ar >= 0), np.flatnonzero(cr >= 0)
        if a_ok.size and c_ok.size:
            block = (matrix[ar[a_ok]] @ matrix[cr[c_ok]].T).toarray()
            sims[np.ix_(a_ok, c_ok)] = block
        keep = a_item[a_lo:a_hi, None] != c_item[None, lo:hi]
        for k in depths:
            kk = keep[:k]
            n = kk.sum(axis=0)
            total = np.where(kk, sims[:k], 0.0).sum(axis=0)
            top = np.where(kk, sims[:k], -np.inf).max(axis=0)
            out[k][0][lo:hi] = np.where(n > 0, total, np.nan)
            out[k][1][lo:hi] = np.where(n > 0, top, np.nan)
    return out


def _inputs(ctx: Context, module, names) -> pl.DataFrame:
    def make() -> pl.DataFrame:
        part = [s for s in module.SPECS if s.name in names]
        return module.build(ctx, part)

    return ctx.memo(f"x_inputs_{module.GROUP}", make)


def build(ctx: Context, specs) -> pl.DataFrame:
    keys = list(KEYS)
    base = ctx.keys.unique().sort(PAIR)
    depths = tuple(sorted({int(s.params[0]) for s in specs if s.signal == "cooc"}))

    def cooc(b, s):
        table = ctx.memo(f"x_cooc_{depths}", lambda: _cooc(ctx, b, depths))
        total, top = table[int(s.params[0])]
        return total if s.agg == "sum" else top

    def side(name):
        if name in _USER_INPUTS:
            table, on = _inputs(ctx, user, _USER_INPUTS), ["uid"]
        else:
            table, on = _inputs(ctx, item, _ITEM_INPUTS), ["item_id"]
        joined = base.select(on).join(table.select(*on, name), on=on, how="left")
        return column(joined, name)

    def gap_pop(b, s):
        lis = side("i_lis_30")
        log_lis = np.log(np.where(lis > 0, lis, np.nan))
        return np.abs(log_lis - side("u_pop_taste"))

    def gap_tl_z(b, s):
        return ratio(side("i_track_len") - side("u_tl_mean_30"), side("u_tl_std_30"))

    rules = {
        ("cooc", "sum"): cooc,
        ("cooc", "max"): cooc,
        ("als", "dot"): lambda b, s: cd.als_dot(_factors(ctx), b).cast(pl.Float64),
        ("item_pop", "gap"): gap_pop,
        ("track_len", "z"): gap_tl_z,
        ("org", "product"): lambda b, s: side("u_org_share_30") * side("i_org_share"),
    }
    return finish(base, keys, specs, rules)


register_group(GROUP, KEYS, 50, build, SPECS)
