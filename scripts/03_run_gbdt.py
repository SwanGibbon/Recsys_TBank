import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import polars as pl

from recsys_tbank import baselines as bl
from recsys_tbank import dataset as ds
from recsys_tbank import features as ft
from recsys_tbank import gbdt as gb
from recsys_tbank.candidates import DEFAULT_CONFIG, LABELS, SLICES
from recsys_tbank.split import DATA_PATH, TEST_TS, VAL_TS, get_window

EXPERIMENTS = {
    "E1a": {"objective": "lambdarank", "label": "y_org", "groups": None},
    "E1b": {"objective": "binary", "label": "y_org", "groups": None},
    "E1src": {
        "objective": "lambdarank",
        "label": "y_org",
        "groups": (("C1", "Z1"), ("C", "Z")),
    },
}
COLUMNS = (
    "ndcg@10",
    "recall@10",
    "recall_cap@10",
    "hit_rate@10",
    "ndcg@100",
    "recall@100",
    "recall_cap@100",
    "hit_rate@100",
    "roc_auc",
    "logloss",
)
SHOW_SLICES = ("all", "novel")
TUNING_NOTE = (
    "train_280 plays the role of the validation window for early stopping, "
    "selection and tuning; val is a pre-registered held-out window used only "
    "for the planned comparisons"
)


def fmt(value) -> str:
    if value is None:
        return "n/a"
    return "nan" if math.isnan(value) else f"{value:.4f}"


def parse_params(text):
    if text is None:
        return dict(gb.DEFAULT_PARAMS)
    raw = text if text.lstrip().startswith("{") else Path(text).read_text()
    return json.loads(raw)


def print_table(record: dict, models) -> None:
    target = record["target_split"]
    print_table_header(record, target)
    print_metric_rows(record, target, models)
    print_all_users_rows(record, target, models)


def print_table_header(record: dict, target: str) -> None:
    n_hist = record["n_hist_users"][target]
    print(
        f"{target} ({record['protocol']} protocol, selection {record['selection_window']}), "
        f"seeds {record['seeds']}, n_hist_users {n_hist}"
    )
    head = f"{'label':6} {'slice':6} {'model':10} " + " ".join(
        f"{c:>14}" for c in COLUMNS
    )
    print(head)


def print_metric_rows(record: dict, target: str, models) -> None:
    for label in LABELS:
        for sl in SHOW_SLICES:
            for m in models:
                row = record["metrics"][target][label][sl][m]
                cells = " ".join(f"{fmt(row.get(c)):>14}" for c in COLUMNS)
                print(f"{label:6} {sl:6} {m:10} {cells}")


def print_all_users_rows(record: dict, target: str, models) -> None:
    print(f"{target}, all users (metric x n_users / n_hist_users), slice all")
    print(
        f"{'label':6} {'model':10} "
        + " ".join(f"{c:>14}" for c in bl.ALL_USERS_METRICS)
    )
    for label in LABELS:
        for m in models:
            row = record["metrics_all_users"][target][label]["all"][m]
            cells = " ".join(f"{fmt(row[c]):>14}" for c in bl.ALL_USERS_METRICS)
            print(f"{label:6} {m:10} {cells}")


class Plan(NamedTuple):
    inner_train: tuple
    sel: str
    final_train: tuple
    target: str
    windows: list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--protocol", choices=gb.PROTOCOLS, default="inner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[gb.SEED])
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--users", type=int, default=0)
    parser.add_argument("--train-windows", nargs="+", default=None)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-perm", action="store_true")
    parser.add_argument("--data", default=str(DATA_PATH))
    return parser


def parse_args(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or args.users < 0:
        parser.error("seeds must be unique and users must be >= 0")
    return args


def resolve_groups(exp: str, spec: dict):
    if spec["groups"] is None:
        return None, None
    found = [o for o in spec["groups"] if all(g in ft.GROUPS for g in o)]
    if not found:
        return None, (
            f"skip {exp}: none of the feature group sets {spec['groups']} "
            f"is registered (registered: {sorted(ft.GROUPS)})"
        )
    return found[0], None


def build_config(args, spec: dict, groups):
    return gb.GBDTConfig(
        objective=spec["objective"],
        label=spec["label"],
        params=parse_params(args.params_json),
        num_boost_round=args.rounds,
        early_stopping_rounds=args.early_stopping,
        groups=groups,
        train_windows=tuple(args.train_windows) if args.train_windows else None,
    )


def make_plan(cfg, protocol: str) -> Plan:
    inner_train, sel, final_train, target = gb.protocol_windows(cfg, protocol)
    windows = list(dict.fromkeys((*inner_train, sel, *final_train, target)))
    return Plan(inner_train, sel, final_train, target, windows)


def load_datas(args, plan: Plan, specs, runtime: dict):
    t0 = time.perf_counter()
    uids = ds.sample_users(args.data, args.users)
    events, info = ds.load_protocol_events(args.protocol, args.data, uids)
    runtime["load"] = time.perf_counter() - t0
    datas = ds.build_dataset(
        events,
        args.protocol,
        plan.windows,
        DEFAULT_CONFIG,
        specs,
        use_cache=not args.no_cache,
    )
    cache = {}
    for name, data in datas.items():
        runtime.update({f"{k}/{name}": v for k, v in data.timings.items()})
        cache[name] = data.cache
    return events, info, datas, cache


def add_seed_scores(frames: dict, key: str, res, target: str) -> None:
    for n, frame in list(frames.items()):
        frames[n] = frame.with_columns(pl.Series(key, res.scores[n]))
    frames[target] = frames[target].with_columns(
        pl.Series(f"p:{label}:{key}", res.probs[label]) for label in LABELS
    )


def fit_seeds(args, cfg, plan: Plan, datas, features, model_keys, runtime: dict):
    frames = {n: gb.eval_frame(datas[n]) for n in (*plan.final_train, plan.target)}
    fits, first = {}, None
    for seed, key in zip(args.seeds, model_keys):
        res = gb.run_protocol(datas, replace(cfg, seed=seed), args.protocol, features)
        runtime.update({f"{k}/s{seed}": v for k, v in res.timings.items()})
        fits[str(seed)] = res.info
        add_seed_scores(frames, key, res, plan.target)
        first = first or res
    return frames, fits, first


def add_baseline_lists(events, datas, frames: dict, runtime: dict) -> dict:
    t0 = time.perf_counter()
    lists = {}
    for n, frame in list(frames.items()):
        frames[n], lists[n] = gb.add_baselines(events, datas[n], frame)
    runtime["baselines"] = time.perf_counter() - t0
    return lists


def evaluate(datas, frames, lists, model_keys, plan: Plan, runtime: dict):
    t0 = time.perf_counter()
    n_hist = {n: datas[n].n_hist_users for n in frames}
    metrics, metrics_all_users = gb.evaluate_models(
        frames,
        {n: datas[n].n_pos for n in frames},
        lists,
        n_hist,
        model_keys,
        plan.target,
        plan.final_train,
    )
    runtime["evaluation"] = time.perf_counter() - t0
    return metrics, metrics_all_users, n_hist


def importance_block(args, cfg, plan: Plan, datas, features, first, runtime: dict):
    t0 = time.perf_counter()
    importance = {"gain": gb.gain_importance(first.fit)}
    if not args.no_perm:
        groups = gb.feature_groups(features)
        importance["permutation_by_group"] = {
            "split": plan.target,
            "label": cfg.label,
            "seed": gb.SEED,
            "values": gb.permutation_importance(
                first.fit,
                datas[plan.target].frame,
                groups,
                ds.group_n_pos(datas, [plan.target], cfg.label),
            ),
        }
    runtime["importance"] = time.perf_counter() - t0
    return importance


def config_block(args, cfg, plan: Plan) -> dict:
    return {
        "gbdt": cfg.config_dict(),
        "inner_train_windows": list(plan.inner_train),
        "final_train_windows": list(plan.final_train),
        "candidates": DEFAULT_CONFIG.config_dict(),
        "baseline_params": gb.BASELINE_PARAMS.config_dict(),
        "baseline_params_source": gb.BASELINE_PARAMS_SOURCE,
        "windows": {
            n: {"T": get_window(n).T, "T_end": get_window(n).T_end}
            for n in plan.windows
        },
        "load_limit": {"inner": VAL_TS, "outer": TEST_TS}[args.protocol],
        "users_sample": {
            "n": args.users,
            "seed": gb.SEED,
            "pool": "uids with timestamp < VAL_TS",
        },
        "protocol_rule": (
            "inner: fit on the inner train windows, early stopping and selection on "
            "train_280 by ndcg@10 of the training label (slice all, n_pos = all window "
            "positives); outer: same inner fit, then refit on inner windows + train_280 "
            f"with num_boost_round = round(best_iteration * {gb.REFIT_FACTOR}) and evaluate on val"
        ),
        "tuning_note": TUNING_NOTE,
        "lambdarank_groups": "(T, uid), contiguous, rows sorted by (T, uid, item_id); groups without a positive of the training label are dropped from lambdarank training only; validation and evaluation keep all rows",
        "binary_rows": "all rows, no reweighting",
        "calibration": (
            "LogLoss on the target split only. binary: own label = model probability, "
            "other label = percentile calibrator; lambdarank: metrics.bin_calibrator "
            f"(bins={gb.CALIB_BINS}) on per-uid percentile ranks of the score; inner: "
            f"fitted on train_280 inner out-of-sample predictions, cross-fitted over "
            f"{gb.CALIB_FOLDS} user folds (seed {gb.SEED}); outer: fitted on all train_280 "
            "inner out-of-sample predictions and applied to val percentiles of the refit model; "
            "baselines: bl._calibrate fitted on the final train windows"
        ),
        "all_users_definition": "metric * n_users / n_hist_users, as in baselines.all_users_metrics",
        "permutation": "rows shuffled jointly for all features of a group over the whole target frame, numpy default_rng(42); delta = ndcg@10 before - after",
    }


def features_block(features) -> dict:
    group_sizes = {g: len(v) for g, v in gb.feature_groups(features).items()}
    return {
        "names": list(features),
        "n": len(features),
        "group_sizes": group_sizes,
        "registered_groups": {
            g: len(grp.specs) for g, grp in sorted(ft.GROUPS.items())
        },
        "feature_version": ft.FEATURE_VERSION,
    }


def dataset_block(datas) -> dict:
    return {
        n: {
            "rows": d.frame.height,
            "users": d.frame["uid"].n_unique(),
            "groups": int(ds.group_sizes(d.frame).size),
            "n_hist_users": d.n_hist_users,
            **{f"pos_{lb}": int(d.frame[lb].sum()) for lb in LABELS},
        }
        for n, d in datas.items()
    }


def n_users_block(metrics, n_hist) -> dict:
    return {
        n: {
            lb: {sl: metrics[n][lb][sl]["B0"]["n_users"] for sl in SLICES}
            for lb in LABELS
        }
        for n in n_hist
    }


def build_record(
    args, cfg, plan: Plan, features, info, datas, fits, importance, results
):
    metrics, metrics_all_users, n_hist = results
    return {
        "seed": args.seeds[0],
        "seeds": list(args.seeds),
        "git_commit": gb.GIT_COMMIT,
        "experiment": args.exp,
        "protocol": args.protocol,
        "selection_window": plan.sel,
        "target_split": plan.target,
        "config": config_block(args, cfg, plan),
        "features": features_block(features),
        "data": info,
        "versions": gb.library_versions(),
        "dataset": dataset_block(datas),
        "fit": fits,
        "importance": importance,
        "n_users": n_users_block(metrics, n_hist),
        "n_hist_users": n_hist,
        "metrics": metrics,
        "metrics_all_users": metrics_all_users,
    }


def finish_record(
    record: dict, model_keys, runtime: dict, start: float, cache: dict
) -> None:
    if len(model_keys) > 1:
        record["seed_stats"] = gb.seed_stats(record["metrics"], model_keys)
    runtime["total"] = time.perf_counter() - start
    record["runtime_seconds"] = runtime
    record["resources"] = {"peak_rss_mb": gb.peak_rss_mb(), "cache_hits": cache}


def print_report(
    record: dict, model_keys, fits: dict, runtime: dict, path, digest
) -> None:
    print_table(record, [*model_keys, *bl.REPORTED_BASELINES])
    for seed, fi in fits.items():
        inner = fi["inner"]
        print(
            f"seed {seed}: best_iteration {inner['best_iteration']} "
            f"inner ndcg@10 {fmt(inner['best_ndcg@10'])} "
            f"train_rows {inner['train_rows']} dropped_groups {inner['dropped_zero_positive_groups']}"
        )
    print("timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in runtime.items()))
    print(f"peak_rss_mb {record['resources']['peak_rss_mb']:.0f}")
    print(f"run log: {path}")
    print(f"metrics_digest: {digest}")


def model_keys_for(exp: str, seeds) -> list:
    return [exp] if len(seeds) == 1 else [f"{exp}:s{s}" for s in seeds]


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = EXPERIMENTS[args.exp]
    groups, skip = resolve_groups(args.exp, spec)
    if skip:
        print(skip)
        return 0
    start = time.perf_counter()
    runtime: dict[str, float] = {}
    cfg = build_config(args, spec, groups)
    plan = make_plan(cfg, args.protocol)
    specs = ft.all_specs()
    features = gb.resolve_features(cfg, specs)
    events, info, datas, cache = load_datas(args, plan, specs, runtime)
    model_keys = model_keys_for(args.exp, args.seeds)
    frames, fits, first = fit_seeds(
        args, cfg, plan, datas, features, model_keys, runtime
    )
    lists = add_baseline_lists(events, datas, frames, runtime)
    results = evaluate(datas, frames, lists, model_keys, plan, runtime)
    importance = importance_block(args, cfg, plan, datas, features, first, runtime)
    record = build_record(
        args, cfg, plan, features, info, datas, fits, importance, results
    )
    finish_record(record, model_keys, runtime, start, cache)
    name = args.name or f"gbdt-{args.exp}"
    path, digest = gb.write_run_log(record, name, Path(args.out_dir))
    print_report(record, model_keys, fits, runtime, path, digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
