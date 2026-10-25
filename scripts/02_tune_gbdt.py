import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from recsys_tbank import dataset as ds
from recsys_tbank import features as ft
from recsys_tbank import gbdt as gb
from recsys_tbank import metrics as mt
from recsys_tbank.candidates import DEFAULT_CONFIG
from recsys_tbank.split import DATA_PATH, VAL_TS

PROTOCOL = "inner"
LABEL = "y_org"
OBJECTIVE = "lambdarank"
SELECTION_METRIC = gb.STOP_METRIC
SEEDS = (42, 43, 44)
TRUNCATION_LEVEL = 30
N_RERUN = 2
FIXED = {"bagging_fraction": 0.8, "bagging_freq": 1, "max_bin": 255}
VARIED = (
    "num_leaves",
    "min_data_in_leaf",
    "feature_fraction",
    "learning_rate",
    "lambda_l2",
)
GRID = (
    {
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "learning_rate": 0.05,
        "lambda_l2": 1.0,
    },
    {
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.6,
        "learning_rate": 0.03,
        "lambda_l2": 10.0,
    },
    {
        "num_leaves": 63,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "learning_rate": 0.03,
        "lambda_l2": 1.0,
    },
    {
        "num_leaves": 15,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "learning_rate": 0.05,
        "lambda_l2": 10.0,
    },
    {
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.6,
        "learning_rate": 0.05,
        "lambda_l2": 10.0,
    },
    {
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.6,
        "learning_rate": 0.03,
        "lambda_l2": 10.0,
    },
    {
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.6,
        "learning_rate": 0.03,
        "lambda_l2": 1.0,
    },
    {
        "num_leaves": 15,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "learning_rate": 0.05,
        "lambda_l2": 1.0,
    },
)
SELECTION_RULE = (
    f"stage 1: fit every grid point with seed {SEEDS[0]} (inner protocol: fit on "
    f"{' + '.join(gb.INNER_TRAIN_WINDOWS)}, early stopping and selection on "
    f"{gb.SELECTION_WINDOW}, metric {SELECTION_METRIC} of {LABEL}, slice all). "
    f"stage 2: re-fit the best {N_RERUN} points (ties by lower point index) and point 0 "
    f"with seeds {list(SEEDS)}; the candidate is the point with the best mean "
    f"{SELECTION_METRIC}. If candidate mean minus point 0 mean is not larger than the "
    "larger of the two seed std values (ddof=1) of the candidate and of point 0, "
    "keep point 0 (the default); otherwise select the candidate. "
    "num_boost_round basis = best_iteration of the selected point at seed 42"
)
TUNING_NOTE = (
    "train_280 plays the role of the validation for tuning, while val is a "
    "pre-registered held-out window used only for the planned comparisons"
)


def point_params(point: int) -> dict:
    return {**GRID[point], **FIXED}


def check_grid(grid=GRID) -> None:
    if len(grid) != 8:
        raise ValueError("grid must have 8 points")
    keys = {tuple(sorted(p)) for p in grid}
    if keys != {tuple(sorted(VARIED))}:
        raise ValueError("every grid point must set exactly the varied parameters")
    if len({tuple(p[k] for k in VARIED) for p in grid}) != len(grid):
        raise ValueError("grid points must be distinct")
    default = {k: gb.DEFAULT_PARAMS[k] for k in VARIED}
    if {k: grid[0][k] for k in VARIED} != default:
        raise ValueError("point 0 must equal the default parameters")
    if {k: gb.DEFAULT_PARAMS[k] for k in FIXED} != FIXED:
        raise ValueError("fixed parameters must equal the defaults")


def rank_points(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda i: (-scores[i], i))


def rerun_points(scores: dict[int, float], n: int = N_RERUN) -> list[int]:
    top = rank_points(scores)[:n]
    return sorted(set(top) | {0})


def mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return float(arr.mean()), std


def choose_point(means: dict[int, float], stds: dict[int, float]) -> tuple[int, dict]:
    best = min(means, key=lambda i: (-means[i], i))
    margin = max(stds[best], stds[0])
    gain = means[best] - means[0]
    keep_default = best == 0 or gain <= margin
    detail = {
        "candidate": best,
        "candidate_mean": means[best],
        "default_mean": means[0],
        "gain_over_default": gain,
        "seed_std_margin": margin,
        "kept_default": keep_default,
    }
    return (0 if keep_default else best), detail


class Context(NamedTuple):
    train: pl.DataFrame
    valid: pl.DataFrame
    n_pos: pl.DataFrame
    scored: pl.DataFrame
    uid_pos: pl.DataFrame
    features: tuple[str, ...]
    rounds: int
    patience: int


def fit_point(ctx: Context, point: int, seed: int) -> tuple[dict, float]:
    cfg = gb.GBDTConfig(
        objective=OBJECTIVE,
        label=LABEL,
        params=point_params(point),
        num_boost_round=ctx.rounds,
        early_stopping_rounds=ctx.patience,
        seed=seed,
        lambdarank_truncation_level=TRUNCATION_LEVEL,
    )
    t0 = time.perf_counter()
    res = gb.fit(ctx.train, ctx.valid, cfg, ctx.n_pos, ctx.features)
    seconds = time.perf_counter() - t0
    scores = gb.predict(res, ctx.valid)
    ev = mt.rank_metrics(
        ctx.scored.with_columns(pl.Series("score", scores)), ctx.uid_pos
    )
    out = {
        "point": point,
        "seed": seed,
        "ndcg@10": res.best_score,
        "ndcg@10_eval": ev["ndcg@10"],
        "recall@10": ev["recall@10"],
        "ndcg@100": ev["ndcg@100"],
        "recall@100": ev["recall@100"],
        "best_iteration": res.best_iteration,
        "rounds_run": len(res.curve),
        "train_rows": res.train_rows,
        "train_groups": res.train_groups,
        "dropped_zero_positive_groups": res.dropped_groups,
    }
    return out, seconds


def fmt(row: dict, seconds: float | None = None) -> str:
    cells = [
        f"{row['ndcg@10']:.4f}",
        f"{row['recall@10']:.4f}",
        f"{row['ndcg@100']:.4f}",
        f"{row['recall@100']:.4f}",
        f"{row['best_iteration']:5d}",
    ]
    if seconds is not None:
        cells.append(f"{seconds:7.1f}s")
    return " ".join(cells)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    check_grid()
    start = time.perf_counter()
    runtime: dict[str, float] = {}
    base = gb.GBDTConfig(
        objective=OBJECTIVE,
        label=LABEL,
        num_boost_round=args.rounds,
        early_stopping_rounds=args.early_stopping,
        lambdarank_truncation_level=TRUNCATION_LEVEL,
    )
    inner_train, sel, _, _ = gb.protocol_windows(base, PROTOCOL)
    windows = [*inner_train, sel]
    specs = ft.all_specs()
    features = gb.resolve_features(base, specs)
    t0 = time.perf_counter()
    events, info = ds.load_protocol_events(PROTOCOL, args.data)
    max_ts = int(events["timestamp"].max())
    if max_ts >= VAL_TS:
        raise RuntimeError(f"loaded events reach {max_ts} >= VAL_TS {VAL_TS}")
    runtime["load"] = time.perf_counter() - t0
    datas = ds.build_dataset(
        events,
        PROTOCOL,
        windows,
        DEFAULT_CONFIG,
        specs,
        use_cache=not args.no_cache,
    )
    runtime["build"] = time.perf_counter() - t0 - runtime["load"]
    cache = {n: d.cache for n, d in datas.items()}
    train = ds.stack_frames([datas[w].frame for w in inner_train])
    valid = datas[sel].frame
    n_pos = ds.group_n_pos(datas, [sel], LABEL)
    ctx = Context(
        train=train,
        valid=valid,
        n_pos=n_pos,
        scored=valid.select("uid", "item_id", pl.col(LABEL).alias("label")),
        uid_pos=datas[sel].n_pos[LABEL].select("uid", pl.col("all").alias("n_pos")),
        features=features,
        rounds=args.rounds,
        patience=args.early_stopping,
    )

    def run(point: int, seed: int):
        row, seconds = fit_point(ctx, point, seed)
        runtime[f"fit/point{point}/s{seed}"] = seconds
        return row, seconds

    print(
        f"protocol {PROTOCOL}, selection {sel} ({SELECTION_METRIC} {LABEL} slice all), "
        f"max loaded timestamp {max_ts} < VAL_TS {VAL_TS}, features {len(features)}"
    )
    print(
        f"{'point':>5} {'params':58} ndcg@10 recall@10 ndcg@100 recall@100 best_it fit"
    )
    results = []
    for point in range(len(GRID)):
        row, seconds = run(point, SEEDS[0])
        results.append(row)
        p = GRID[point]
        shown = (
            f"nl{p['num_leaves']} mdl{p['min_data_in_leaf']} ff{p['feature_fraction']} "
            f"lr{p['learning_rate']} l2 {p['lambda_l2']:g}"
        )
        print(f"{point:5d} {shown:58} {fmt(row, seconds)}", flush=True)
    stage1 = {r["point"]: r["ndcg@10"] for r in results}
    chosen = rerun_points(stage1)
    reeval = {}
    for point in chosen:
        rows = [results[point]]
        for seed in SEEDS[1:]:
            row, seconds = run(point, seed)
            rows.append(row)
            print(f"point {point} seed {seed}: {fmt(row, seconds)}", flush=True)
        mean, std = mean_std([r["ndcg@10"] for r in rows])
        reeval[point] = {
            "seeds": list(SEEDS),
            "per_seed": rows,
            "mean_ndcg@10": mean,
            "std_ndcg@10": std,
            "mean_best_iteration": float(np.mean([r["best_iteration"] for r in rows])),
        }
    means = {p: v["mean_ndcg@10"] for p, v in reeval.items()}
    stds = {p: v["std_ndcg@10"] for p, v in reeval.items()}
    selected, detail = choose_point(means, stds)
    basis = reeval[selected]["per_seed"][0]["best_iteration"]
    selected_params = point_params(selected)
    for p in chosen:
        print(
            f"point {p}: mean {SELECTION_METRIC} {means[p]:.5f} std {stds[p]:.5f} "
            f"mean best_iteration {reeval[p]['mean_best_iteration']:.1f}"
        )
    print(
        f"selected point {selected} (candidate {detail['candidate']}, gain over default "
        f"{detail['gain_over_default']:.5f}, margin {detail['seed_std_margin']:.5f}, "
        f"kept default {detail['kept_default']}), num_boost_round basis {basis}"
    )
    runtime["total"] = time.perf_counter() - start
    record = {
        "seed": SEEDS[0],
        "seeds": list(SEEDS),
        "git_commit": gb.GIT_COMMIT,
        "experiment": "tune-gbdt",
        "protocol": PROTOCOL,
        "selection_window": sel,
        "selection_metric": f"{SELECTION_METRIC} ({LABEL}, slice all, n_pos = all window positives)",
        "tuning_note": TUNING_NOTE,
        "objective": OBJECTIVE,
        "label": LABEL,
        "inner_train_windows": list(inner_train),
        "early_stopping_rounds": args.early_stopping,
        "num_boost_round_max": args.rounds,
        "lambdarank_truncation_level": TRUNCATION_LEVEL,
        "fixed_params": FIXED,
        "grid": [point_params(i) for i in range(len(GRID))],
        "results": results,
        "reeval": {str(p): v for p, v in reeval.items()},
        "reeval_points": chosen,
        "selection_rule": SELECTION_RULE,
        "selection_detail": detail,
        "selected_point": selected,
        "selected_params": selected_params,
        "selected_num_boost_round_basis": basis,
        "selected_num_boost_round_basis_source": (
            f"best_iteration of point {selected} at seed {SEEDS[0]} on {sel}"
        ),
        "max_timestamp_loaded": max_ts,
        "val_ts": VAL_TS,
        "features": {
            "names": list(features),
            "n": len(features),
            "feature_version": ft.FEATURE_VERSION,
        },
        "candidates": DEFAULT_CONFIG.config_dict(),
        "data": info,
        "versions": gb.library_versions(),
        "dataset": {
            n: {
                "rows": d.frame.height,
                "users": d.frame["uid"].n_unique(),
                "n_hist_users": d.n_hist_users,
            }
            for n, d in datas.items()
        },
        "runtime_seconds": runtime,
        "resources": {"peak_rss_mb": gb.peak_rss_mb(), "cache_hits": cache},
    }
    out_dir = Path(args.out_dir)
    path, digest = gb.write_run_log(record, "tune-gbdt", out_dir)
    best = out_dir / "gbdt-best-params.json"
    best.write_text(json.dumps(selected_params, indent=2, sort_keys=True) + "\n")
    print("timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in runtime.items()))
    print(f"peak_rss_mb {record['resources']['peak_rss_mb']:.0f}")
    print(f"run log: {path}")
    print(f"best params: {best}")
    print(f"metrics_digest: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
