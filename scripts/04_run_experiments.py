import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl

from recsys_tbank import baselines as bl
from recsys_tbank import dataset as ds
from recsys_tbank import features as ft
from recsys_tbank import gbdt as gb
from recsys_tbank import metrics as mt
from recsys_tbank.candidates import DEFAULT_CONFIG, LABELS, SLICES
from recsys_tbank.split import DATA_PATH, TEST_TS, get_window

EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E5", "IMP", "BOOT")
LOAD_PROTOCOL = "outer"
SEEDS = (42, 43, 44, 45, 46)
FALLBACK_SEEDS = (3,)
REF = f"E1a:s{gb.SEED}"
DEFAULT_PARAMS_JSON = Path("runs") / "gbdt-best-params.json"
DEFAULT_BASELINE_LOG = Path("runs") / "20261018-210235-baselines.json"
COMPARED_BASELINES = ("B0", "B1", "B2", "B3")
SRC_GROUPS = ("C", "Z")
E3_WINDOWS = (7, 30, 90)
E3_VARIANTS = ("w7", "w30", "w90", "no_decay")
NONE_WINDOW_RULES = ("all", "keep")
WINDOWLESS_GROUPS = ("C", "Z")
STATIC_SIGNALS = (
    ("item", "track_len"),
    ("item", "artist"),
    ("item", "album"),
    ("item", "mapping"),
)
DELTA_METRICS = (
    "ndcg@10",
    "recall@10",
    "recall_cap@10",
    "hit_rate@10",
    "ndcg@100",
    "recall@100",
    "recall_cap@100",
)
SELECT_LABEL = "y_org"
SELECT_SLICE = "all"
SELECT_METRIC = gb.STOP_METRIC
TOP_GAIN = 30
N_BOOT = 1000
CI_LEVEL = 95.0
BOOT_K = 10
BOOT_LABEL = "y_org"
BOOT_SLICES = ("all", "novel")
BOOT_METRICS = (f"ndcg@{BOOT_K}", f"recall_cap@{BOOT_K}")
BOOT_PAIRS = ((REF, "B3"), (REF, "B5"), (REF, "E1src"))
TABLE_COLUMNS = (
    "ndcg@10",
    "recall@10",
    "recall_cap@10",
    "ndcg@100",
    "recall@100",
    "roc_auc",
    "logloss",
)
SHOW_SLICES = ("all", "novel")
NOT_SIGNIFICANT = "not significant"
EXCEEDS_STD = "exceeds seed std"
NO_STD = "n/a"
INNER_PROB_METHOD = (
    "percentile bin_calibrator cross-fitted over user folds on train_280 "
    "(lambdarank, or the other label of a binary model); binary own label = model probability"
)
TUNING_NOTE = (
    "train_280 plays the role of the validation for tuning, early stopping and every "
    "selection, while val is a pre-registered held-out window used only for the planned comparisons"
)
PROTOCOL_RULE = (
    "inner: fit on train_266 + train_273, early stopping on train_280 by ndcg@10 of the "
    "training label (slice all), metrics on train_280; outer: same inner fit, then refit on "
    "train_266 + train_273 + train_280 with num_boost_round = round(best_iteration * "
    f"{gb.REFIT_FACTOR}) and frozen params, metrics on val"
)
INNER_DERIVATION = (
    "events are loaded once with timestamp < TEST_TS (test window never loaded); inner numbers "
    "come from the inner fit of gbdt.run_protocol (train rows of train_266 and train_273, early "
    "stopping on train_280), which reads no val row, so they equal a separate inner-protocol run"
)
SELECTION_POLICY = (
    f"every choice (best E3 window variant, most important E2 group) uses inner numbers on "
    f"{gb.SELECTION_WINDOW} only: {SELECT_METRIC} of {SELECT_LABEL}, slice {SELECT_SLICE}, "
    f"rounded to {bl.DIGITS} decimals, ties to the earlier variant; val numbers are reported, never used to choose"
)
SIGNIFICANCE_RULE = (
    f"delta = variant - {REF} (same seed); seed_std = std (ddof=1) of the metric over the E1a "
    f"seeds in the same protocol, split, label and slice; '{NOT_SIGNIFICANT}' when |delta| < "
    f"seed_std, '{EXCEEDS_STD}' otherwise, '{NO_STD}' when either is missing"
)
E3_RULE = {
    "all": (
        "a spec with window w (days) is kept in variant wW when w <= W; a spec with window None "
        "that aggregates history (counts, rates, recency, tenure, decay, co-listen, ALS, artist "
        "and album history) has an unbounded window and is kept only in variant all; specs "
        f"without a time window (groups {list(WINDOWLESS_GROUPS)}, candidate pool outputs, and "
        f"static item attributes with (entity, signal) in {[list(s) for s in STATIC_SIGNALS]}) "
        "are kept in every variant; no_decay drops every spec whose agg contains 'decay'"
    ),
    "keep": (
        "a spec with window w (days) is kept in variant wW when w <= W; every spec with window "
        "None is kept in every variant; no_decay drops every spec whose agg contains 'decay'"
    ),
}
E4_RULE = (
    f"frozen params and the E1a round count rule: every depth is refit without early stopping "
    f"with the num_boost_round of {REF} outer (round(best_iteration * {gb.REFIT_FACTOR}) of its "
    "inner fit); depth 3 is the E1a outer model itself; LogLoss uses the percentile calibrator "
    f"fitted on the {REF} inner out-of-sample predictions on train_280; inner is not defined for "
    "depth 1, so only outer numbers are reported"
)
BOOT_RULE = (
    f"paired user bootstrap on val, label {BOOT_LABEL}: users with >= 1 positive of the slice; "
    f"per-user ndcg@{BOOT_K} and recall_cap@{BOOT_K} exactly as metrics.rank_metrics; delta = "
    f"model a - model b per user; {N_BOOT} resamples of users with replacement, numpy "
    f"default_rng({gb.SEED}) restarted for every comparison, so both models and every "
    f"comparison of a slice share the same user samples; percentile {CI_LEVEL:g}% CI"
)


def fmt(value) -> str:
    if value is None:
        return "n/a"
    return "nan" if math.isnan(value) else f"{value:.4f}"


def seeds_for(fallback: int | None) -> tuple[int, ...]:
    if fallback is None:
        return SEEDS
    if fallback not in FALLBACK_SEEDS:
        raise ValueError(f"seed fallback must be one of {FALLBACK_SEEDS}")
    return SEEDS[:fallback]


def load_params(path: str | Path) -> tuple[dict, dict]:
    path = Path(path)
    if not path.is_file():
        warning = (
            f"WARNING: params file {path} not found; the experiments use the code defaults "
            "gbdt.DEFAULT_PARAMS instead of tuned params"
        )
        return dict(gb.DEFAULT_PARAMS), {
            "path": str(path),
            "found": False,
            "used": "gbdt.DEFAULT_PARAMS",
            "warning": warning,
        }
    raw = path.read_bytes()
    params = json.loads(raw)
    if isinstance(params, Mapping) and "selected_params" in params:
        params = params["selected_params"]
    if not isinstance(params, Mapping):
        raise TypeError(f"params file {path} must hold a JSON object")
    params = dict(params)
    gb.GBDTConfig(params=params)
    return params, {
        "path": str(path),
        "found": True,
        "used": "params file",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def effective_window(spec: ft.FeatureSpec, none_rule: str = "all"):
    if none_rule not in NONE_WINDOW_RULES:
        raise ValueError(f"none_rule must be one of {NONE_WINDOW_RULES}")
    if spec.window is not None:
        return spec.window
    if none_rule == "keep" or spec.group in WINDOWLESS_GROUPS:
        return None
    if (spec.entity, spec.signal) in STATIC_SIGNALS:
        return None
    return "all"


def keeps_window(spec: ft.FeatureSpec, days: int, none_rule: str = "all") -> bool:
    w = effective_window(spec, none_rule)
    return w is None or (w != "all" and w <= days)


def window_variants(
    specs: Sequence[ft.FeatureSpec], none_rule: str = "all"
) -> dict[str, tuple[str, ...]]:
    out = {
        f"w{d}": tuple(s.name for s in specs if keeps_window(s, d, none_rule))
        for d in E3_WINDOWS
    }
    out["no_decay"] = tuple(s.name for s in specs if "decay" not in s.agg)
    out["all"] = tuple(s.name for s in specs)
    empty = [k for k, v in out.items() if not v]
    if empty:
        raise ValueError(f"empty E3 variants: {empty}")
    return out


def significance(delta, std) -> str:
    if delta is None or std is None:
        return NO_STD
    if not (math.isfinite(delta) and math.isfinite(std)):
        return NO_STD
    return NOT_SIGNIFICANT if abs(delta) < std else EXCEEDS_STD


def delta_table(
    by_label: Mapping,
    ref: str,
    keys: Sequence[str],
    stats: Mapping,
    metric_names: Sequence[str] = DELTA_METRICS,
) -> dict:
    out = {}
    for key in keys:
        out[key] = {}
        for lb, by_slice in by_label.items():
            out[key][lb] = {}
            for sl, by_model in by_slice.items():
                cells = {}
                for m in metric_names:
                    value, base = by_model[key].get(m), by_model[ref].get(m)
                    std = stats.get(lb, {}).get(sl, {}).get(m, {}).get("std")
                    delta = None if value is None or base is None else value - base
                    cells[m] = {
                        "value": value,
                        "reference": base,
                        "delta": delta,
                        "seed_std": std,
                        "verdict": significance(delta, std),
                    }
                out[key][lb][sl] = cells
    return out


def select_inner(values: Mapping[str, float], window: str) -> str:
    gb.check_selection_window(window)
    return bl.select_best(list(values.items()))


def selection_record(
    kind: str, values: Mapping[str, float], window: str, rule: str
) -> dict:
    return {
        "kind": kind,
        "protocol": "inner",
        "window": gb.check_selection_window(window),
        "label": SELECT_LABEL,
        "slice": SELECT_SLICE,
        "metric": SELECT_METRIC,
        "values": dict(values),
        "chosen": select_inner(values, window),
        "rule": rule,
    }


def per_user_metrics(
    scored: pl.DataFrame, n_pos: pl.DataFrame, k: int = BOOT_K
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if (
        scored.height
        and scored.select(
            pl.col("score").is_null().any() | pl.col("score").is_nan().any()
        ).item()
    ):
        raise ValueError("score has null or NaN")
    n_pos = mt._check_frames(scored, n_pos)
    users = n_pos.filter(pl.col("n_pos") >= 1).sort("uid")
    ranked = (
        scored.join(users.select("uid"), on="uid", how="semi")
        .sort(["uid", "score", "item_id"], descending=[False, True, False])
        .with_columns(rank=pl.int_range(1, pl.len() + 1).over("uid"))
    )
    gain = pl.col("label").cast(pl.Float64)
    top = pl.col("rank") <= k
    per_user = (
        users.join(
            ranked.group_by("uid").agg(
                (gain * top).sum().alias("hits"),
                (gain * top / (pl.col("rank") + 1).log(2)).sum().alias("dcg"),
            ),
            on="uid",
            how="left",
        )
        .fill_null(0)
        .sort("uid")
    )
    npos = per_user["n_pos"].to_numpy()
    hits = per_user["hits"].to_numpy()
    dcg = per_user["dcg"].to_numpy()
    ideal = np.concatenate(([0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))))
    capped = np.minimum(npos, k)
    return per_user["uid"].to_numpy(), {
        f"ndcg@{k}": dcg / np.where(capped > 0, ideal[capped], 1.0),
        f"recall_cap@{k}": hits / np.maximum(capped, 1),
    }


def paired_bootstrap(a, b, n_boot: int = N_BOOT, seed: int = gb.SEED) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired vectors must be 1-d and of equal length")
    out = {"n_users": int(a.size), "n_boot": n_boot, "seed": seed, "ci_level": CI_LEVEL}
    if a.size == 0:
        return {
            **out,
            **dict.fromkeys(
                ("mean_a", "mean_b", "mean_delta", "boot_std", "ci_low", "ci_high")
            ),
            "ci_excludes_zero": None,
        }
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, d.size, d.size)].mean()
    tail = (100.0 - CI_LEVEL) / 2.0
    lo, hi = np.percentile(boots, [tail, 100.0 - tail])
    return {
        **out,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_delta": float(d.mean()),
        "boot_std": float(boots.std(ddof=1)) if n_boot > 1 else None,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def compare_baseline_log(metrics: Mapping, log: Mapping, keys=COMPARED_BASELINES):
    mine = bl._clean(metrics)
    theirs = log.get("metrics", {})
    mismatches = []
    for split, by_label in mine.items():
        for lb, by_slice in by_label.items():
            for sl, by_model in by_slice.items():
                for key in keys:
                    a = by_model.get(key)
                    b = theirs.get(split, {}).get(lb, {}).get(sl, {}).get(key)
                    if a != b:
                        mismatches.append(f"{split}/{lb}/{sl}/{key}")
    return {
        "equal": not mismatches,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
    }


@dataclass
class Model:
    key: str
    cfg: gb.GBDTConfig
    features: tuple[str, ...]
    result: gb.ProtocolResult | None
    final: gb.FitResult
    info: dict
    timings: dict
    scores: dict = field(default_factory=dict)
    probs: dict = field(default_factory=dict)


class Runner:
    def __init__(self, datas, base: gb.GBDTConfig, specs, events=None, seeds=SEEDS):
        self.datas = datas
        self.base = base
        self.specs = tuple(specs)
        self.events = events
        self.seeds = tuple(seeds)
        inner, sel, final, target = gb.protocol_windows(base, "outer")
        if "test" in (*final, target):
            raise PermissionError("test windows are never used")
        self.inner_train = inner
        self.sel = sel
        self.final_train = final
        self.target = target
        self.splits = {"inner": (*inner, sel), "outer": (*final, target)}
        self.calib = {"inner": inner, "outer": final}
        self.targets = {"inner": sel, "outer": target}
        self.models: dict[str, Model] = {}
        self.memo: dict = {}
        self.baseline_frames: dict = {}
        self.lists: dict = {}
        self.clock: dict[str, float] = {}
        self.reused: set[str] = set()

    def tick(self, name: str, t0: float) -> None:
        self.clock[name] = self.clock.get(name, 0.0) + time.perf_counter() - t0

    def start(self) -> None:
        self.clock, self.reused = {}, set()

    def _cached(self, key: str) -> Model | None:
        if key in self.models:
            self.reused.add(key)
        return self.models.get(key)

    def _remember(self, model: Model) -> Model:
        self.models[model.key] = model
        return model

    def fit(self, key: str, **overrides) -> Model:
        if (cached := self._cached(key)) is not None:
            return cached
        cfg = replace(self.base, **overrides)
        features = gb.resolve_features(cfg, self.specs)
        t0 = time.perf_counter()
        res = gb.run_protocol(self.datas, cfg, "outer", features)
        self.tick(f"fit/{key}", t0)
        info = {
            "inner": res.info["inner"],
            "outer": res.info["outer"],
            "probability": {
                "outer": res.info["probability"],
                "inner": INNER_PROB_METHOD,
            },
        }
        model = Model(key, cfg, features, res, res.fit, info, dict(res.timings))
        model.scores.update({("outer", s): v for s, v in res.scores.items()})
        model.probs.update({("outer", lb): v for lb, v in res.probs.items()})
        return self._remember(model)

    def fit_fixed(self, key: str, base: Model, windows: Sequence[str]) -> Model:
        if (cached := self._cached(key)) is not None:
            return cached
        rounds = base.info["outer"]["num_boost_round"]
        cfg = replace(base.cfg, num_boost_round=rounds, early_stopping_rounds=0)
        t0 = time.perf_counter()
        full = ds.stack_frames([self.datas[w].frame for w in windows])
        final = gb.fit(full, None, cfg, None, base.features)
        del full
        fit_s = time.perf_counter() - t0
        self.tick(f"fit/{key}", t0)
        t0 = time.perf_counter()
        frame = self.datas[self.target].frame
        scores = gb.predict(final, frame)
        calib_frame = self.datas[self.sel].frame
        calib_scores = self.score(base, "inner", self.sel)
        probs, methods = {}, {}
        for lb in LABELS:
            probs[lb], methods[lb] = gb.to_probability(
                base.cfg, lb, frame, scores, calib_frame, calib_scores
            )
        info = {
            "inner": None,
            "outer": {
                "train_windows": list(windows),
                "num_boost_round": rounds,
                "rounds_source": f"{base.key} outer num_boost_round",
                "train_rows": final.train_rows,
                "train_groups": final.train_groups,
                "dropped_zero_positive_groups": final.dropped_groups,
            },
            "probability": {"outer": methods},
        }
        timings = {"fit_outer": fit_s, "predict_outer": time.perf_counter() - t0}
        model = Model(key, cfg, base.features, None, final, info, timings)
        model.scores[("outer", self.target)] = scores
        model.probs.update({("outer", lb): p for lb, p in probs.items()})
        return self._remember(model)

    def score(self, model: Model, protocol: str, split: str) -> np.ndarray:
        k = (protocol, split)
        if k not in model.scores:
            if protocol != "inner" or model.result is None:
                raise KeyError(f"{model.key} has no {protocol} scores on {split}")
            t0 = time.perf_counter()
            model.scores[k] = gb.predict(
                model.result.inner_fit, self.datas[split].frame
            )
            model.timings[f"predict_inner/{split}"] = time.perf_counter() - t0
        return model.scores[k]

    def prob(self, model: Model, protocol: str, label: str) -> np.ndarray:
        k = (protocol, label)
        if k not in model.probs:
            if protocol != "inner" or model.result is None:
                raise KeyError(f"{model.key} has no {protocol} probabilities")
            model.probs[k], _ = gb.to_probability(
                model.cfg,
                label,
                self.datas[self.sel].frame,
                self.score(model, "inner", self.sel),
            )
        return model.probs[k]

    def baseline_frame(self, split: str) -> pl.DataFrame:
        if split not in self.baseline_frames:
            if self.events is None:
                raise ValueError("baselines need the loaded events")
            t0 = time.perf_counter()
            frame, top = gb.add_baselines(
                self.events, self.datas[split], gb.eval_frame(self.datas[split])
            )
            self.baseline_frames[split], self.lists[split] = frame, top
            self.tick("baselines", t0)
        return self.baseline_frames[split]

    def _evaluate_split(self, protocol: str, split: str, keys: Sequence[str]) -> None:
        calib, target = self.calib[protocol], self.targets[protocol]
        models = [k for k in keys if k not in bl.REPORTED_BASELINES]
        scored = [k for k in keys if k in gb.SCORED_BASELINES]
        needs_base = bool(scored) or "B4" in keys
        frame = (
            self.baseline_frame(split)
            if needs_base
            else gb.eval_frame(self.datas[split])
        )
        t0 = time.perf_counter()
        cols = [
            pl.Series(k, self.score(self.models[k], protocol, split)) for k in models
        ]
        if split == target:
            cols += [
                pl.Series(f"p:{lb}:{k}", self.prob(self.models[k], protocol, lb))
                for k in models
                for lb in LABELS
            ]
        frame = frame.with_columns(cols)
        if scored and split == target:
            frames = {w: self.baseline_frame(w) for w in calib}
            frames[target] = frame
            bl._calibrate(frames, scored, calib, target)
            frame = frames[target]
        grid = {split: {lb: {sl: {} for sl in SLICES} for lb in LABELS}}
        n_pos = {split: self.datas[split].n_pos}
        bl._evaluate_keys({split: frame}, n_pos, [*models, *scored], target, grid)
        if "B4" in keys:
            bl._evaluate_catalog({split: self.lists[split]}, n_pos, target, grid)
        for k in keys:
            self.memo[(protocol, split, k)] = {
                lb: {sl: grid[split][lb][sl][k] for sl in SLICES} for lb in LABELS
            }
        self.tick("evaluation", t0)

    def evaluate(self, protocol: str, keys: Sequence[str], splits: Sequence[str]):
        bad = [s for s in splits if s not in self.splits[protocol]]
        if bad:
            raise ValueError(f"splits {bad} are not part of the {protocol} protocol")
        for split in splits:
            need = [k for k in keys if (protocol, split, k) not in self.memo]
            if need:
                self._evaluate_split(protocol, split, need)
        metrics = {
            split: {
                lb: {
                    sl: {k: self.memo[(protocol, split, k)][lb][sl] for k in keys}
                    for sl in SLICES
                }
                for lb in LABELS
            }
            for split in splits
        }
        all_users = {
            split: {
                lb: {
                    sl: {
                        k: bl.all_users_metrics(
                            metrics[split][lb][sl][k], self.datas[split].n_hist_users
                        )
                        for k in keys
                    }
                    for sl in SLICES
                }
                for lb in LABELS
            }
            for split in splits
        }
        return metrics, all_users

    def e1a(self) -> list[str]:
        return [self.fit(f"E1a:s{s}", seed=s).key for s in self.seeds]

    def ref(self) -> Model:
        return self.fit(REF, seed=gb.SEED)

    def registered_groups(self) -> list[str]:
        groups = {s.group for s in self.specs}
        known = [
            g.name for g in sorted(ft.GROUPS.values(), key=lambda g: (g.order, g.name))
        ]
        return [g for g in known if g in groups] + sorted(groups - set(known))


def model_section(runner: Runner, keys: Sequence[str]) -> dict:
    out = {}
    for k in keys:
        if k not in runner.models:
            continue
        m = runner.models[k]
        out[k] = {
            "config": m.cfg.config_dict(),
            "n_features": len(m.features),
            "features": list(m.features),
            "fit": m.info,
        }
    return out


def base_record(
    runner: Runner, meta: Mapping, exp: str, protocols: Sequence[str]
) -> dict:
    return {
        "seed": gb.SEED,
        "seeds": list(runner.seeds),
        "git_commit": gb.GIT_COMMIT,
        "experiment": exp,
        "protocol": "+".join(protocols),
        "protocols": list(protocols),
        "selection_window": gb.check_selection_window(runner.sel),
        "target_split": {p: runner.targets[p] for p in protocols},
        "config": dict(meta["config"]),
        "features": dict(meta["features"]),
        "data": meta["data"],
        "versions": meta["versions"],
        "dataset": meta["dataset"],
        "metrics_layout": "metrics[protocol][split][label][slice][model] -> metric values",
    }


def finish_record(runner: Runner, meta: Mapping, record: dict, keys, t0: float) -> dict:
    record["models"] = model_section(runner, keys)
    runtime = dict(meta["runtime"])
    runtime.update({f"exp/{k}": v for k, v in runner.clock.items()})
    for k in keys:
        if k in runner.models:
            runtime.update(
                {f"model/{k}/{s}": v for s, v in runner.models[k].timings.items()}
            )
    runtime["exp/total"] = time.perf_counter() - t0
    record["runtime_seconds"] = runtime
    record["resources"] = {
        "peak_rss_mb": gb.peak_rss_mb(),
        "peak_rss_scope": "whole process up to the end of this experiment",
        "cache_hits": meta["cache"],
        "reused_fits": sorted(runner.reused),
    }
    return record


def baseline_log_check(meta: Mapping, metrics: Mapping) -> dict:
    log_path = meta["baseline_log"]
    check: dict = {"path": str(log_path), "baselines": list(COMPARED_BASELINES)}
    if meta["users"]:
        check["status"] = "skipped: user subsample, the run log covers all dev users"
    elif not Path(log_path).is_file():
        check["status"] = "skipped: run log not found"
    else:
        log = json.loads(Path(log_path).read_text())
        if log.get("data", {}).get("path") != meta["data"].get("path"):
            check["status"] = "skipped: run log was built on another data file"
        else:
            check.update(compare_baseline_log(metrics, log))
            check["status"] = "equal" if check["equal"] else "MISMATCH"
    return check


def exp_e0(runner: Runner, meta: Mapping) -> dict:
    t0 = time.perf_counter()
    runner.start()
    keys = list(bl.REPORTED_BASELINES)
    splits = runner.splits["outer"]
    metrics, all_users = runner.evaluate("outer", keys, splits)
    record = base_record(runner, meta, "E0", ["outer"])
    check = baseline_log_check(meta, metrics)
    record.update(
        {
            "selection": "none: baseline params frozen from Phase 1 (see config.baseline_params_source)",
            "baseline_log_check": check,
            "metrics": {"outer": metrics},
            "metrics_all_users": {"outer": all_users},
        }
    )
    return finish_record(runner, meta, record, [], t0)


def exp_e1(runner: Runner, meta: Mapping) -> dict:
    t0 = time.perf_counter()
    runner.start()
    e1a = runner.e1a()
    keys = [*e1a, runner.fit("E1b", objective="binary").key]
    groups = runner.registered_groups()
    src = None
    if all(g in groups for g in SRC_GROUPS):
        src = runner.fit("E1src", groups=SRC_GROUPS).key
        keys.append(src)
    shown = [*keys, *bl.REPORTED_BASELINES]
    record = base_record(runner, meta, "E1", ["inner", "outer"])
    metrics, all_users, stats = {}, {}, {}
    for p in ("inner", "outer"):
        metrics[p], all_users[p] = runner.evaluate(p, shown, runner.splits[p])
        stats[p] = gb.seed_stats(metrics[p], e1a) if len(e1a) > 1 else {}
    record.update(
        {
            "variants": {
                "E1a": "lambdarank, label y_org, all features, seeds "
                + str(list(runner.seeds)),
                "E1b": "binary, label y_org, all features, seed 42",
                "E1src": f"lambdarank, label y_org, groups {list(SRC_GROUPS)} only, seed 42"
                if src
                else f"skipped: groups {list(SRC_GROUPS)} not registered",
            },
            "e1a_keys": e1a,
            "splits_note": "inner: train_266/train_273 in-sample, train_280 out-of-sample; outer: train windows in-sample, val out-of-sample",
            "metrics": metrics,
            "metrics_all_users": all_users,
            "seed_stats": stats,
        }
    )
    return finish_record(runner, meta, record, keys, t0)


def _deltas(runner: Runner, e1a, variants, ref: str = REF):
    metrics, all_users, stats, deltas = {}, {}, {}, {}
    for p in ("inner", "outer"):
        target = runner.targets[p]
        metrics[p], all_users[p] = runner.evaluate(p, [*e1a, *variants], [target])
        stats[p] = gb.seed_stats(metrics[p], e1a) if len(e1a) > 1 else {}
        deltas[p] = {
            target: delta_table(
                metrics[p][target], ref, variants, stats[p].get(target, {})
            )
        }
    return metrics, all_users, stats, deltas


def _inner_value(runner: Runner, metrics: Mapping, key: str) -> float:
    return metrics["inner"][runner.sel][SELECT_LABEL][SELECT_SLICE][key][SELECT_METRIC]


def exp_e2(runner: Runner, meta: Mapping) -> dict:
    t0 = time.perf_counter()
    runner.start()
    e1a = runner.e1a()
    groups = runner.registered_groups()
    variants = {g: runner.fit(f"E2:-{g}", exclude_groups=(g,)).key for g in groups}
    metrics, all_users, stats, deltas = _deltas(runner, e1a, list(variants.values()))
    ref_value = _inner_value(runner, metrics, REF)
    drops = {
        g: ref_value - _inner_value(runner, metrics, k) for g, k in variants.items()
    }
    record = base_record(runner, meta, "E2", ["inner", "outer"])
    record.update(
        {
            "groups": groups,
            "variants": variants,
            "reference": REF,
            "e1a_keys": e1a,
            "significance_rule": SIGNIFICANCE_RULE,
            "selection": selection_record(
                "most important group = largest inner drop when the group is left out",
                drops,
                runner.sel,
                SELECTION_POLICY,
            ),
            "metrics": metrics,
            "metrics_all_users": all_users,
            "seed_stats": stats,
            "deltas": deltas,
        }
    )
    return finish_record(runner, meta, record, [*e1a, *variants.values()], t0)


def exp_e3(runner: Runner, meta: Mapping) -> dict:
    t0 = time.perf_counter()
    runner.start()
    rule = meta["e3_none_window"]
    sets = window_variants(runner.specs, rule)
    e1a = runner.e1a()
    keys = {v: runner.fit(f"E3:{v}", features=sets[v]).key for v in E3_VARIANTS}
    keys["all"] = REF
    metrics, all_users, stats, deltas = _deltas(
        runner, e1a, [keys[v] for v in E3_VARIANTS]
    )
    values = {v: _inner_value(runner, metrics, k) for v, k in keys.items()}
    full = set(sets["all"])
    record = base_record(runner, meta, "E3", ["inner", "outer"])
    record.update(
        {
            "mapping_rule": {"none_window": rule, "text": E3_RULE[rule]},
            "effective_windows": {
                s.name: effective_window(s, rule) for s in runner.specs
            },
            "variant_features": {
                v: {
                    "n": len(sets[v]),
                    "kept": list(sets[v]),
                    "dropped": sorted(full - set(sets[v])),
                }
                for v in (*E3_VARIANTS, "all")
            },
            "variants": keys,
            "reference": REF,
            "e1a_keys": e1a,
            "significance_rule": SIGNIFICANCE_RULE,
            "selection": selection_record(
                "best feature-window variant by inner value",
                values,
                runner.sel,
                SELECTION_POLICY,
            ),
            "metrics": metrics,
            "metrics_all_users": all_users,
            "seed_stats": stats,
            "deltas": deltas,
        }
    )
    return finish_record(runner, meta, record, [*e1a, *keys.values()], t0)


def exp_e4(runner: Runner, meta: Mapping) -> dict:
    t0 = time.perf_counter()
    runner.start()
    ref = runner.ref()
    windows = {
        1: (runner.sel,),
        2: (runner.inner_train[-1], runner.sel),
        3: runner.final_train,
    }
    depths = {
        d: runner.fit_fixed(f"E4:d{d}", ref, w).key for d, w in windows.items() if d < 3
    }
    depths[3] = ref.key
    keys = [depths[d] for d in sorted(depths)]
    metrics, all_users = runner.evaluate("outer", keys, [runner.target])
    record = base_record(runner, meta, "E4", ["outer"])
    record.update(
        {
            "rule": E4_RULE,
            "depths": {
                str(d): {"model": depths[d], "train_windows": list(windows[d])}
                for d in sorted(depths)
            },
            "num_boost_round": ref.info["outer"]["num_boost_round"],
            "inner": "not defined for depth 1; not reported",
            "metrics": {"outer": metrics},
            "metrics_all_users": {"outer": all_users},
        }
    )
    return finish_record(runner, meta, record, keys, t0)


def exp_e5(runner: Runner, meta: Mapping) -> dict:
    t0 = time.perf_counter()
    runner.start()
    ref = runner.ref()
    alt = runner.fit("E5:y_all", label="y_all")
    keys = [ref.key, alt.key]
    metrics, all_users = runner.evaluate("outer", keys, [runner.target])
    record = base_record(runner, meta, "E5", ["outer"])
    record.update(
        {
            "train_label": {ref.key: "y_org", alt.key: "y_all"},
            "rule": "lambdarank, all features; early stopping on train_280 uses the training label; every model is evaluated on both labels",
            "metrics": {"outer": metrics},
            "metrics_all_users": {"outer": all_users},
        }
    )
    return finish_record(runner, meta, record, keys, t0)


def exp_imp(runner: Runner, meta: Mapping) -> dict:
    t0 = time.perf_counter()
    runner.start()
    ref = runner.ref()
    gain = gb.gain_importance(ref.final)
    top = list(gain["by_feature"])[:TOP_GAIN]
    frame = runner.datas[runner.target].frame
    n_pos = ds.group_n_pos(runner.datas, [runner.target], SELECT_LABEL)
    t1 = time.perf_counter()
    by_group = gb.permutation_importance(
        ref.final, frame, gb.feature_groups(ref.features), n_pos, SELECT_LABEL, gb.SEED
    )
    by_feature = gb.permutation_importance(
        ref.final, frame, {f: [f] for f in top}, n_pos, SELECT_LABEL, gb.SEED
    )
    runner.tick("permutation", t1)
    record = base_record(runner, meta, "IMP", ["outer"])
    record.update(
        {
            "model": ref.key,
            "gain": {"source": f"{ref.key} outer refit, total gain", **gain},
            "permutation": {
                "split": runner.target,
                "label": SELECT_LABEL,
                "metric": f"delta {SELECT_METRIC} = before - after, n_pos = all window positives",
                "seed": gb.SEED,
                "rule": "rows shuffled jointly for all columns of the entry over the whole val frame, numpy default_rng(42)",
                "by_group": by_group,
                "top_features_by_gain": top,
                "by_feature": by_feature,
            },
        }
    )
    return finish_record(runner, meta, record, [ref.key], t0)


def exp_boot(runner: Runner, meta: Mapping) -> dict:
    t0 = time.perf_counter()
    runner.start()
    ref = runner.ref()
    models = {ref.key: ref}
    groups = runner.registered_groups()
    pairs = list(BOOT_PAIRS)
    if all(g in groups for g in SRC_GROUPS):
        models["E1src"] = runner.fit("E1src", groups=SRC_GROUPS)
    else:
        pairs = [p for p in pairs if p[1] != "E1src"]
    split = runner.target
    base = runner.baseline_frame(split)
    t1 = time.perf_counter()
    columns = {k: runner.score(m, "outer", split) for k, m in models.items()}
    columns.update({b: base[b].to_numpy() for b in ("B3", "B5")})
    data = runner.datas[split]
    frame = data.frame.select(
        "uid", "item_id", "novel", pl.col(BOOT_LABEL).alias("label")
    )
    keys = list(columns)
    metrics, _ = runner.evaluate("outer", keys, [split])
    out, checks = {}, {}
    for sl in BOOT_SLICES:
        n_pos = data.n_pos[BOOT_LABEL].select("uid", pl.col(sl).alias("n_pos"))
        row_filter = bl.SLICE_FILTERS[sl]
        vectors, uids = {}, None
        for k, score in columns.items():
            scored = frame.with_columns(pl.Series("score", score))
            if row_filter is not None:
                scored = scored.filter(row_filter)
            u, vec = per_user_metrics(
                scored.select("uid", "item_id", "label", "score"), n_pos
            )
            if uids is not None and not np.array_equal(u, uids):
                raise ValueError("per-user vectors are not aligned")
            uids, vectors[k] = u, vec
            checks[f"{sl}/{k}"] = {
                m: abs(float(np.mean(vec[m])) - metrics[split][BOOT_LABEL][sl][k][m])
                if vec[m].size
                else None
                for m in BOOT_METRICS
            }
        out[sl] = {
            f"{a} vs {b}": {
                m: paired_bootstrap(vectors[a][m], vectors[b][m]) for m in BOOT_METRICS
            }
            for a, b in pairs
        }
    runner.tick("bootstrap", t1)
    record = base_record(runner, meta, "BOOT", ["outer"])
    record.update(
        {
            "rule": BOOT_RULE,
            "split": split,
            "label": BOOT_LABEL,
            "pairs": [f"{a} vs {b}" for a, b in pairs],
            "bootstrap": out,
            "per_user_check": {
                "rule": "abs(mean of the per-user vector - aggregate metric of the run)",
                "values": checks,
            },
        }
    )
    return finish_record(runner, meta, record, list(models), t0)


RUNNERS = {
    "E0": exp_e0,
    "E1": exp_e1,
    "E2": exp_e2,
    "E3": exp_e3,
    "E4": exp_e4,
    "E5": exp_e5,
    "IMP": exp_imp,
    "BOOT": exp_boot,
}


def print_metrics(record: Mapping) -> None:
    for p, by_split in record.get("metrics", {}).items():
        target = record["target_split"][p]
        print(
            f"[{record['experiment']}] {p} protocol, split {target}, "
            f"selection {record['selection_window']}"
        )
        print(
            f"{'label':6} {'slice':6} {'model':12} "
            + " ".join(f"{c:>13}" for c in TABLE_COLUMNS)
        )
        for lb in LABELS:
            for sl in SHOW_SLICES:
                for model, row in by_split[target][lb][sl].items():
                    cells = " ".join(f"{fmt(row.get(c)):>13}" for c in TABLE_COLUMNS)
                    print(f"{lb:6} {sl:6} {model:12} {cells}")


def print_e0_check(record: Mapping) -> None:
    if record["experiment"] == "E0":
        print(f"baseline log check: {record['baseline_log_check'].get('status')}")


def print_deltas(record: Mapping) -> None:
    if "deltas" not in record:
        return
    print(
        f"deltas vs {record['reference']} ({SELECT_METRIC}, {SELECT_LABEL}, slice {SELECT_SLICE})"
    )
    for p, by_split in record["deltas"].items():
        for split, by_key in by_split.items():
            for key, cells in by_key.items():
                c = cells[SELECT_LABEL][SELECT_SLICE][SELECT_METRIC]
                print(
                    f"  {p:5} {split:9} {key:14} delta {fmt(c['delta'])} "
                    f"std {fmt(c['seed_std'])} {c['verdict']}"
                )


def print_selection(record: Mapping) -> None:
    if "selection" in record and isinstance(record["selection"], Mapping):
        s = record["selection"]
        print(f"selection ({s['protocol']} {s['window']}): {s['chosen']}")


def print_permutation(record: Mapping) -> None:
    if record["experiment"] == "IMP":
        perm = record["permutation"]["by_group"]
        print(
            "permutation by group (delta ndcg@10 on val): "
            + ", ".join(f"{g} {fmt(v['delta'])}" for g, v in perm.items())
        )


def print_bootstrap(record: Mapping) -> None:
    if record["experiment"] != "BOOT":
        return
    for sl, by_pair in record["bootstrap"].items():
        for pair, by_metric in by_pair.items():
            for m, r in by_metric.items():
                print(
                    f"  {sl:6} {pair:20} {m:14} delta {fmt(r['mean_delta'])} "
                    f"CI [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] n {r['n_users']}"
                )


EXTRA_PRINTERS = (
    print_e0_check,
    print_deltas,
    print_selection,
    print_permutation,
    print_bootstrap,
)


def print_extra(record: Mapping) -> None:
    for printer in EXTRA_PRINTERS:
        printer(record)


def build_meta(args, runner, info, datas, params_source, runtime, cache) -> dict:
    seeds = list(runner.seeds)
    return {
        "users": args.users,
        "baseline_log": args.baseline_log,
        "e3_none_window": args.e3_none_window,
        "runtime": runtime,
        "cache": cache,
        "data": info,
        "versions": gb.library_versions(),
        "dataset": {
            n: {
                "rows": d.frame.height,
                "users": d.frame["uid"].n_unique(),
                "groups": int(ds.group_sizes(d.frame).size),
                "n_hist_users": d.n_hist_users,
                **{f"pos_{lb}": int(d.frame[lb].sum()) for lb in LABELS},
            }
            for n, d in datas.items()
        },
        "features": {
            "registered": [s.name for s in runner.specs],
            "n_registered": len(runner.specs),
            "registered_groups": {
                g: sum(s.group == g for s in runner.specs)
                for g in runner.registered_groups()
            },
            "feature_version": ft.FEATURE_VERSION,
        },
        "config": {
            "gbdt_base": runner.base.config_dict(),
            "params_source": params_source,
            "e1a_seeds": seeds,
            "seed_policy": "5 seeds 42-46 (amendment 5)"
            if len(seeds) == len(SEEDS)
            else f"fallback to {len(seeds)} seeds {seeds[0]}-{seeds[-1]} (amendment 5)",
            "inner_train_windows": list(runner.inner_train),
            "final_train_windows": list(runner.final_train),
            "windows": {
                n: {"T": get_window(n).T, "T_end": get_window(n).T_end} for n in datas
            },
            "load_limit": TEST_TS,
            "load_protocol": LOAD_PROTOCOL,
            "users_sample": {
                "n": args.users,
                "seed": gb.SEED,
                "pool": "uids with timestamp < VAL_TS",
            },
            "candidates": DEFAULT_CONFIG.config_dict(),
            "baseline_params": gb.BASELINE_PARAMS.config_dict(),
            "baseline_params_source": gb.BASELINE_PARAMS_SOURCE,
            "protocol_rule": PROTOCOL_RULE,
            "inner_derivation": INNER_DERIVATION,
            "selection_policy": SELECTION_POLICY,
            "tuning_note": TUNING_NOTE,
            "lambdarank_groups": "(T, uid), contiguous; groups without a positive of the training label are dropped from lambdarank training only",
            "calibration": (
                "LogLoss on the target split only; models: inner = percentile calibrator cross-fitted "
                "over user folds on train_280, outer = calibrator fitted on train_280 inner "
                "out-of-sample predictions (binary own label = model probability); baselines: "
                "bl._calibrate fitted on the protocol's train windows"
            ),
            "all_users_definition": "metric * n_users / n_hist_users, as in baselines.all_users_metrics",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", nargs="+", choices=[*EXPERIMENTS, "ALL"], required=True
    )
    parser.add_argument("--params-json", default=str(DEFAULT_PARAMS_JSON))
    parser.add_argument(
        "--seeds-fallback", type=int, choices=FALLBACK_SEEDS, default=None
    )
    parser.add_argument("--users", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--e3-none-window", choices=NONE_WINDOW_RULES, default="all")
    parser.add_argument("--baseline-log", default=str(DEFAULT_BASELINE_LOG))
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--cache-dir", default=str(ds.CACHE_DIR))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--data", default=str(DATA_PATH))
    args = parser.parse_args(argv)
    if args.users < 0:
        parser.error("users must be >= 0")
    exps = list(EXPERIMENTS) if "ALL" in args.exp else list(dict.fromkeys(args.exp))
    start = time.perf_counter()
    params, source = load_params(args.params_json)
    if not source["found"]:
        banner = "!" * 78
        for stream in (sys.stderr, sys.stdout):
            print(f"{banner}\n{source['warning']}\n{banner}", file=stream, flush=True)
    base = gb.GBDTConfig(
        params=params,
        num_boost_round=args.rounds,
        early_stopping_rounds=args.early_stopping,
    )
    _, _, final, target = gb.protocol_windows(base, LOAD_PROTOCOL)
    windows = list(dict.fromkeys((*final, target)))
    if "test" in windows:
        raise PermissionError("test windows are never loaded")
    specs = ft.all_specs()
    runtime: dict[str, float] = {}
    t0 = time.perf_counter()
    uids = ds.sample_users(args.data, args.users)
    events, info = ds.load_protocol_events(LOAD_PROTOCOL, args.data, uids)
    runtime["shared/load"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    datas = ds.build_dataset(
        events,
        LOAD_PROTOCOL,
        windows,
        DEFAULT_CONFIG,
        specs,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )
    runtime["shared/build"] = time.perf_counter() - t0
    cache = {}
    for name, data in datas.items():
        runtime.update({f"shared/{k}/{name}": v for k, v in data.timings.items()})
        cache[name] = data.cache
    runner = Runner(datas, base, specs, events, seeds_for(args.seeds_fallback))
    meta = build_meta(args, runner, info, datas, source, runtime, cache)
    for exp in exps:
        record = RUNNERS[exp](runner, meta)
        path, digest = gb.write_run_log(record, f"exp-{exp}", Path(args.out_dir))
        print_metrics(record)
        print_extra(record)
        rt = record["runtime_seconds"]
        print(
            f"[{exp}] experiment wall {rt['exp/total']:.1f} s, peak_rss_mb "
            f"{record['resources']['peak_rss_mb']:.0f}, run log {path}"
        )
        print(f"[{exp}] metrics_digest: {digest}", flush=True)
    if not source["found"]:
        print(source["warning"], file=sys.stderr)
    print(f"total wall {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
