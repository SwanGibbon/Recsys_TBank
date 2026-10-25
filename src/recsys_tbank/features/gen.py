import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real

import numpy as np
import polars as pl

from recsys_tbank.features import meta
from recsys_tbank.features.prep import (
    PAIR,
    SMOOTH_M,
    Context,
    aligned,
    column,
    decay,
    global_rate,
    in_window,
    ratio,
    smoothed,
    to_series,
    ts_counts,
)
from recsys_tbank.features.spec import (
    GROUPS,
    FeatureSpec,
    register_group,
    window_suffix,
)
from recsys_tbank.split import DAY

ENTITIES = ("user", "item", "pair", "user_artist", "user_album", "artist", "album")
LISTEN_SIGNALS = ("L", "P", "PO", "org", "skip")
EVENT_SIGNALS = ("like", "dislike", "unlike", "undislike")
NUMERIC_SIGNALS = ("ratio", "tl")
SIGNALS = (*LISTEN_SIGNALS, *EVENT_SIGNALS, *NUMERIC_SIGNALS)
WINDOWS = (1, 3, 7, 14, 30, 60, 90, 180, None)
COARSE_WINDOWS = (7, 30, 90, None)
TREND_SHORT = {30: 7, 90: 30, 180: 30, None: 90}
DECAY_WINDOWS = (90, None)
HALF_LIVES = (3, 7, 14, 30, 90)
AGGS = (
    "count",
    "rate_s",
    "share_user",
    "trend",
    "recency",
    "age_first",
    "active_days",
    "nunique",
    "decay",
    "mean",
    "max",
    "std",
    "state",
)
GENERATED_GROUPS = ("GU", "GI", "GD", "GA")
ENTITY_GROUP = {
    "user": "GU",
    "item": "GI",
    "pair": "GD",
    "user_artist": "GA",
    "user_album": "GA",
    "artist": "GA",
    "album": "GA",
}
USER_SIDE = ("user", "pair", "user_artist", "user_album")
PAIR_LIKE = ("pair", "user_artist", "user_album")
VOLUME = ("L", "P", "PO")
EVENT_AGGS = ("count", "recency", "state", "rate_s")
AGG_SIGNALS = {
    "count": (*LISTEN_SIGNALS, *EVENT_SIGNALS),
    "rate_s": ("P", "PO", "org", "skip", *EVENT_SIGNALS),
    "share_user": LISTEN_SIGNALS,
    "trend": VOLUME,
    "recency": (*LISTEN_SIGNALS, *EVENT_SIGNALS),
    "age_first": VOLUME,
    "active_days": VOLUME,
    "nunique": VOLUME,
    "decay": VOLUME,
    "mean": NUMERIC_SIGNALS,
    "max": ("ratio",),
    "std": NUMERIC_SIGNALS,
    "state": ("like", "dislike"),
}
AGG_WINDOWS = {
    "count": WINDOWS,
    "rate_s": COARSE_WINDOWS,
    "share_user": COARSE_WINDOWS,
    "trend": tuple(TREND_SHORT),
    "recency": (None,),
    "age_first": (None,),
    "active_days": COARSE_WINDOWS,
    "nunique": COARSE_WINDOWS,
    "decay": DECAY_WINDOWS,
    "mean": COARSE_WINDOWS,
    "max": COARSE_WINDOWS,
    "std": COARSE_WINDOWS,
    "state": (None,),
}

_GROUP_KEYS = {"GU": ("uid",), "GI": ("item_id",), "GD": tuple(PAIR), "GA": tuple(PAIR)}
_GROUP_ORDER = {"GU": 110, "GI": 120, "GD": 130, "GA": 140}
_KEYS = {
    "user": ("uid",),
    "item": ("item_id",),
    "pair": tuple(PAIR),
    "user_artist": ("uid", "artist_id"),
    "user_album": ("uid", "album_id"),
    "artist": ("artist_id",),
    "album": ("album_id",),
}
_OWNER = {
    "user_artist": "artist",
    "user_album": "album",
    "artist": "artist",
    "album": "album",
}
_OTHER = {
    "user": "item_id",
    "item": "uid",
    "user_artist": "item_id",
    "user_album": "item_id",
    "artist": "uid",
    "album": "uid",
}
_FLAG = {"L": None, "P": "pos", "PO": "orgpos", "org": "org", "skip": "skip"}
_RATE_DEN = {"P": "L", "PO": "org", "org": "L", "skip": "L"}
_STATE_OFF = {"like": "unlike", "dislike": "undislike"}
_SUFFIX = {"decay": "_hl", "trend": "_s"}
_FIELDS = ("entity", "signal", "window", "agg")


def _is_window(window) -> bool:
    return window is None or type(window) is int


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _fmt(value) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _name(entity, signal, window, agg, params) -> str:
    name = f"g_{entity}_{signal}_w{window_suffix(window)}_{agg}"
    if agg in _SUFFIX and params:
        name += _SUFFIX[agg] + "_".join(_fmt(p) for p in params)
    return name


def _default_params(agg, window) -> tuple[float, ...]:
    if agg == "rate_s":
        return (float(SMOOTH_M),)
    if agg == "trend" and window in TREND_SHORT:
        return (float(TREND_SHORT[window]),)
    return ()


def make_spec(
    entity: str,
    signal: str,
    window: int | None,
    agg: str,
    params: Sequence[float] | None = None,
) -> FeatureSpec:
    params = _default_params(agg, window) if params is None else tuple(params)
    params = tuple(float(p) for p in params)
    return FeatureSpec(
        _name(entity, signal, window, agg, params),
        ENTITY_GROUP.get(entity, ""),
        entity,
        signal,
        window,
        agg,
        params,
    )


def spec_to_dict(spec: FeatureSpec) -> dict:
    return {
        "entity": spec.entity,
        "signal": spec.signal,
        "window": spec.window,
        "agg": spec.agg,
        "params": list(spec.params),
    }


def _identity(s: FeatureSpec) -> bool:
    return (
        s.entity in ENTITIES
        and s.signal in SIGNALS
        and s.agg in AGGS
        and _is_window(s.window)
        and isinstance(s.params, tuple)
        and all(_is_number(p) for p in s.params)
    )


def _params_ok(s: FeatureSpec) -> bool:
    if s.agg == "decay":
        return len(s.params) == 1 and s.params[0] in HALF_LIVES
    return s.params == _default_params(s.agg, s.window)


RULES: tuple[tuple[str, Callable[[FeatureSpec], bool]], ...] = (
    (
        "event_window_agg",
        lambda s: (
            s.signal not in EVENT_SIGNALS
            or (s.window in COARSE_WINDOWS and s.agg in EVENT_AGGS)
        ),
    ),
    (
        "tl_static",
        lambda s: (
            s.signal != "tl" or (s.entity not in PAIR_LIKE and s.agg in ("mean", "std"))
        ),
    ),
    ("share_user_entity", lambda s: s.agg != "share_user" or s.entity in USER_SIDE),
    (
        "trend_decay_signal",
        lambda s: s.agg not in ("trend", "decay") or s.signal in VOLUME,
    ),
    ("nunique_entity", lambda s: s.agg != "nunique" or s.entity != "pair"),
    ("state_entity", lambda s: s.agg != "state" or s.entity in PAIR_LIKE),
    ("signal_agg", lambda s: s.signal in AGG_SIGNALS[s.agg]),
    ("agg_window", lambda s: s.window in AGG_WINDOWS[s.agg]),
    (
        "user_share_L",
        lambda s: (
            not (s.entity == "user" and s.agg == "share_user" and s.signal == "L")
        ),
    ),
    (
        "user_rate_listen",
        lambda s: (
            not (
                s.entity == "user" and s.agg == "rate_s" and s.signal in LISTEN_SIGNALS
            )
        ),
    ),
    (
        "item_tl",
        lambda s: (
            not (s.entity == "item" and s.signal == "tl")
            or (s.agg == "mean" and s.window is None)
        ),
    ),
    ("params", _params_ok),
    (
        "name_group",
        lambda s: (
            s.group == ENTITY_GROUP[s.entity]
            and s.name == _name(s.entity, s.signal, s.window, s.agg, s.params)
        ),
    ),
)
RULE_NAMES = ("identity", *(name for name, _ in RULES))


def violations(spec: FeatureSpec) -> list[str]:
    if not isinstance(spec, FeatureSpec) or not _identity(spec):
        return ["identity"]
    return [name for name, rule in RULES if not rule(spec)]


def legal(spec: FeatureSpec) -> bool:
    return not violations(spec)


def enumerate_specs(entities: Iterable[str] | None = None) -> list[FeatureSpec]:
    chosen = ENTITIES if entities is None else tuple(entities)
    unknown = sorted(set(chosen) - set(ENTITIES))
    if unknown:
        raise ValueError(f"unknown entities: {unknown}")
    out = []
    for entity in ENTITIES:
        if entity not in chosen:
            continue
        for signal in SIGNALS:
            for agg in AGGS:
                for window in WINDOWS:
                    options = [(h,) for h in HALF_LIVES] if agg == "decay" else [None]
                    for params in options:
                        spec = make_spec(entity, signal, window, agg, params)
                        if legal(spec):
                            out.append(spec)
    return out


def spec_count(specs: Iterable[FeatureSpec] | None = None) -> dict:
    specs = enumerate_specs() if specs is None else list(specs)
    by_entity = {e: sum(s.entity == e for s in specs) for e in ENTITIES}
    by_agg = {a: sum(s.agg == a for s in specs) for a in AGGS}
    return {
        "total": len(specs),
        "by_entity": {k: v for k, v in by_entity.items() if v},
        "by_agg": {k: v for k, v in by_agg.items() if v},
    }


def parse_spec(obj) -> FeatureSpec:
    if isinstance(obj, str | bytes):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as exc:
            raise ValueError(f"spec is not valid JSON: {exc}") from exc
    if not isinstance(obj, Mapping):
        raise TypeError(f"spec must be a JSON object, got {type(obj).__name__}")
    unknown = sorted(set(obj) - {*_FIELDS, "params"})
    missing = [f for f in _FIELDS if f not in obj]
    if unknown or missing:
        raise ValueError(f"spec fields: unknown {unknown}, missing {missing}")
    entity, signal, agg = obj["entity"], obj["signal"], obj["agg"]
    if not all(isinstance(v, str) for v in (entity, signal, agg)):
        raise ValueError("entity, signal and agg must be strings")
    window = None if obj["window"] == "all" else obj["window"]
    if not _is_window(window):
        raise ValueError(f"window must be an integer, null or 'all', got {window!r}")
    params = obj.get("params")
    if params is not None:
        if not isinstance(params, list | tuple) or not all(
            _is_number(p) for p in params
        ):
            raise ValueError(f"params must be a list of numbers, got {params!r}")
        if not params:
            params = None
    spec = make_spec(entity, signal, window, agg, params)
    bad = violations(spec)
    if bad:
        raise ValueError(f"illegal feature spec {spec.name}: {bad}")
    return spec


def _cond(T: int, signal: str, window: int | None) -> pl.Expr:
    c = in_window(T, window)
    flag = _FLAG[signal]
    return c if flag is None else c & pl.col(flag)


def _ts_of(signal: str) -> pl.Expr:
    flag = _FLAG[signal]
    return pl.col("ts") if flag is None else pl.col("ts").filter(pl.col(flag))


def _lis_count(T, signal, window):
    alias = f"n_{signal}_{window_suffix(window)}"
    return "lis", alias, _cond(T, signal, window).sum().alias(alias)


def _ev_count(T, signal, window):
    alias = f"e_{signal}_{window_suffix(window)}"
    expr = ((pl.col("event_type") == signal) & in_window(T, window)).sum()
    return "ev", alias, expr.alias(alias)


def _numeric(T, signal, window, kind):
    ws = window_suffix(window)
    alias = f"{kind}_{signal}_{ws}"
    value = pl.col(signal)
    c = in_window(T, window)
    exprs = {
        "sum": value.filter(c).sum(),
        "sq": (value * value).filter(c).sum(),
        "nn": value.filter(c).count(),
        "max": value.filter(c).max(),
    }
    return "lis", alias, exprs[kind].alias(alias)


def _requests(T: int, entity: str, s: FeatureSpec) -> list[tuple[str, str, pl.Expr]]:
    sig, w = s.signal, s.window
    ws = window_suffix(w)
    if s.agg == "count":
        return [_ev_count(T, sig, w) if sig in EVENT_SIGNALS else _lis_count(T, sig, w)]
    if s.agg == "rate_s":
        if sig in EVENT_SIGNALS:
            return [_ev_count(T, sig, w), _lis_count(T, "L", w)]
        return [_lis_count(T, sig, w), _lis_count(T, _RATE_DEN[sig], w)]
    if s.agg == "share_user":
        return [_lis_count(T, sig, w)]
    if s.agg == "trend":
        return [_lis_count(T, sig, w), _lis_count(T, sig, int(s.params[0]))]
    if s.agg == "recency":
        if sig in EVENT_SIGNALS:
            expr = pl.col("ts").filter(pl.col("event_type") == sig).max()
            return [("ev", f"elast_{sig}", expr.alias(f"elast_{sig}"))]
        return [("lis", f"last_{sig}", _ts_of(sig).max().alias(f"last_{sig}"))]
    if s.agg == "age_first":
        return [("lis", f"first_{sig}", _ts_of(sig).min().alias(f"first_{sig}"))]
    if s.agg == "active_days":
        expr = pl.col("day").filter(_cond(T, sig, w)).n_unique()
        return [("lis", f"days_{sig}_{ws}", expr.alias(f"days_{sig}_{ws}"))]
    if s.agg == "nunique":
        expr = pl.col(_OTHER[entity]).filter(_cond(T, sig, w)).n_unique()
        return [("lis", f"uniq_{sig}_{ws}", expr.alias(f"uniq_{sig}_{ws}"))]
    if s.agg == "mean":
        return [_numeric(T, sig, w, k) for k in ("sum", "nn")]
    if s.agg == "std":
        return [_numeric(T, sig, w, k) for k in ("sum", "sq", "nn")]
    if s.agg == "max":
        return [_numeric(T, sig, w, "max")]
    return []


def _owner_key(entity: str) -> str:
    return f"{_OWNER[entity]}_id"


def _listens(ctx: Context, entity: str) -> pl.DataFrame:
    if entity not in _OWNER:
        return ctx.listens
    key = _owner_key(entity)
    return ctx.memo(
        f"gen_listens_{key}",
        lambda: meta._listens(ctx).filter(pl.col(key).is_not_null()),
    )


def _events(ctx: Context, entity: str) -> pl.DataFrame:
    if entity not in _OWNER:
        return ctx.events
    key = _owner_key(entity)

    def make() -> pl.DataFrame:
        owners = ctx.memo(
            "gen_owner_events",
            lambda: ctx.events.join(
                meta._items(ctx).select("item_id", "artist_id", "album_id"),
                on="item_id",
                how="left",
            ),
        )
        return owners.filter(pl.col(key).is_not_null())

    return ctx.memo(f"gen_events_{key}", make)


def _family(signal: str) -> str:
    return "ev" if signal in EVENT_SIGNALS else "lis"


def _present(ctx: Context, entity: str, family: str, base: pl.DataFrame) -> np.ndarray:
    keys = list(_KEYS[entity])
    if entity in _OWNER:
        return base[keys[-1]].is_not_null().to_numpy()
    pair_listen = entity == "pair" and family == "lis"
    source = ctx.listens if pair_listen else ctx.events
    tag = "lis" if pair_listen else "any"

    def make() -> pl.DataFrame:
        targets = base.unique()
        return (
            source.select(keys)
            .join(targets, on=keys, how="semi")
            .unique()
            .with_columns(pl.lit(True).alias("_has"))
        )

    table = ctx.memo(f"gen_has_{entity}_{tag}", make)
    joined = base.join(table, on=keys, how="left", maintain_order="left")
    return joined["_has"].is_not_null().to_numpy()


def _fill(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    return np.where(present, np.nan_to_num(values, nan=0.0), np.nan)


def _decay_table(ctx, entity, spec, targets) -> pl.DataFrame:
    keys = list(_KEYS[entity])
    flag = _FLAG[spec.signal]

    def make() -> pl.DataFrame:
        sub = _listens(ctx, entity).join(targets, on=keys, how="semi")
        if flag is not None:
            sub = sub.filter(pl.col(flag))
        return ts_counts(sub, keys)

    counts = ctx.memo(f"gen_ts_{entity}_{spec.signal}", make)
    return decay(counts, keys, ctx.T, spec.params[0], spec.window)


def _state_table(ctx, entity, spec, targets) -> pl.DataFrame:
    on, off = spec.signal, _STATE_OFF[spec.signal]
    et = pl.col("event_type")

    def pair_state(scope) -> pl.DataFrame:
        return (
            ctx.events.filter(et.is_in([on, off]))
            .join(scope, on=scope.columns, how="semi")
            .with_columns((et == off).cast(pl.UInt8).alias("_tie"))
            .group_by(PAIR)
            .agg(
                (et.sort_by(["ts", "_tie"]).last() == on).cast(pl.Int64).alias("value")
            )
        )

    if entity == "pair":
        return pair_state(targets)
    key = _owner_key(entity)
    return (
        pair_state(targets.select("uid").unique())
        .filter(pl.col("value") == 1)
        .join(meta._items(ctx).select("item_id", key), on="item_id", how="inner")
        .filter(pl.col(key).is_not_null())
        .group_by(["uid", key])
        .agg(pl.len().cast(pl.Int64).alias("value"))
    )


def _entity_columns(
    ctx: Context, entity: str, specs: Sequence[FeatureSpec], out: pl.DataFrame
) -> dict[str, pl.Series]:
    T = ctx.T
    keys = list(_KEYS[entity])
    base = out.select(keys)
    targets = base.drop_nulls().unique()
    wanted: dict[str, dict[str, pl.Expr]] = {"lis": {}, "ev": {}}
    for s in specs:
        for table, alias, expr in _requests(T, entity, s):
            wanted[table].setdefault(alias, expr)
    frames = {}
    for table, exprs in wanted.items():
        if not exprs:
            continue
        data = _listens(ctx, entity) if table == "lis" else _events(ctx, entity)
        stats = (
            data.join(targets, on=keys, how="semi").group_by(keys).agg(*exprs.values())
        )
        frames[table] = base.join(stats, on=keys, how="left", maintain_order="left")
    present_memo: dict[str, np.ndarray] = {}

    def present(family):
        if family not in present_memo:
            present_memo[family] = _present(ctx, entity, family, base)
        return present_memo[family]

    def get(request):
        table, alias, _ = request
        return column(frames[table], alias)

    totals: dict[str, dict[str, float]] = {}
    for table in ("lis", "ev"):
        exprs = {
            alias: expr
            for s in specs
            if s.agg == "rate_s"
            for t, alias, expr in _requests(T, entity, s)
            if t == table
        }
        if exprs:
            data = _listens(ctx, entity) if table == "lis" else _events(ctx, entity)
            row = data.select(*exprs.values()).row(0, named=True)
            totals[table] = {k: float(v) for k, v in row.items()}

    def total(request) -> float:
        table, alias, _ = request
        return totals[table][alias]

    def value(s):
        fam = present(_family(s.signal))
        reqs = _requests(T, entity, s)
        if s.agg in ("count", "active_days", "nunique"):
            return _fill(get(reqs[0]), fam)
        if s.agg == "rate_s":
            k, n = (_fill(get(r), fam) for r in reqs)
            prior = global_rate(np.array([total(reqs[0])]), np.array([total(reqs[1])]))
            return smoothed(k, n, prior, s.params[0])
        if s.agg == "share_user":
            users = ctx.user_listens(s.window).rename({"n": "value"})
            den = aligned(out.select("uid"), users, ["uid"])
            return ratio(_fill(get(reqs[0]), fam), den)
        if s.agg == "trend":
            long_len = s.window if s.window is not None else T / DAY
            k_long, k_short = (_fill(get(r), fam) for r in reqs)
            return ratio(k_short * long_len, k_long * int(s.params[0]))
        if s.agg in ("recency", "age_first"):
            return (T - get(reqs[0])) / DAY
        if s.agg == "mean":
            return ratio(*(get(r) for r in reqs))
        if s.agg == "std":
            total_, sq, n = (get(r) for r in reqs)
            mean = ratio(total_, n)
            var = ratio(sq - total_ * mean, n - 1)
            return np.sqrt(np.maximum(var, 0.0))
        if s.agg == "max":
            return get(reqs[0])
        if s.agg == "decay":
            return _fill(
                aligned(base, _decay_table(ctx, entity, s, targets), keys), fam
            )
        state = aligned(base, _state_table(ctx, entity, s, targets), keys)
        return state if entity == "pair" else _fill(state, fam)

    return {s.name: to_series(s.name, value(s)).cast(pl.Float32) for s in specs}


def _out_frame(ctx: Context, group: str) -> pl.DataFrame:
    if group in ("GU", "GI"):
        key = _GROUP_KEYS[group][0]
        return ctx.keys.select(key).unique().sort(key)
    pairs = ctx.keys.unique().sort(PAIR)
    if group == "GD":
        return pairs
    mapping = meta._items(ctx).select("item_id", "artist_id", "album_id")
    return pairs.join(mapping, on="item_id", how="left", maintain_order="left")


def _builder(group: str) -> Callable[[Context, Sequence[FeatureSpec]], pl.DataFrame]:
    def build(ctx: Context, specs: Sequence[FeatureSpec]) -> pl.DataFrame:
        specs = tuple(specs)
        for s in specs:
            bad = violations(s)
            if bad or s.group != group:
                raise ValueError(f"spec {s.name} is not a legal {group} spec: {bad}")
        out = _out_frame(ctx, group)
        cols: dict[str, pl.Series] = {}
        for entity in ENTITIES:
            part = [s for s in specs if s.entity == entity]
            if part:
                cols.update(_entity_columns(ctx, entity, part, out))
        keys = list(_GROUP_KEYS[group])
        return out.select(keys).with_columns(cols[s.name] for s in specs)

    return build


_BUILDERS = {g: _builder(g) for g in GENERATED_GROUPS}


def register_generated() -> None:
    for name in GENERATED_GROUPS:
        group = GROUPS.get(name)
        if group is None:
            register_group(
                name, _GROUP_KEYS[name], _GROUP_ORDER[name], _BUILDERS[name], ()
            )
        elif group.build is not _BUILDERS[name]:
            raise ValueError(f"group {name!r} is registered by another module")


def unregister_generated() -> None:
    for name in GENERATED_GROUPS:
        group = GROUPS.get(name)
        if group is not None and group.build is _BUILDERS[name]:
            GROUPS.pop(name)
