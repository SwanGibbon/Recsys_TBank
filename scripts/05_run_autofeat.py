import argparse
import ast
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import psutil

from recsys_tbank import autofeat as af
from recsys_tbank import baselines as bl
from recsys_tbank import candidates as cd
from recsys_tbank import dataset as ds
from recsys_tbank import features as ft
from recsys_tbank import gbdt as gb
from recsys_tbank import split as sp
from recsys_tbank.candidates import DEFAULT_CONFIG, LABELS, SLICES
from recsys_tbank.features import cache as fc
from recsys_tbank.features import gen
from recsys_tbank.split import DATA_PATH, DAY, VAL_TS, history

STAGES = ("build", "screen", "select", "confirm", "val")
DEFAULT_PARAMS_JSON = Path("runs") / "gbdt-best-params.json"
R0_KEY = f"E1a:s{gb.SEED}"
R0_TOL = 1e-6
MIN_FREE_GB = 10.0
GIB = 1024**3
DISK_EXIT_CODE = 2
R0_EXIT_CODE = 1
STORAGE = (
    "one zstd parquet file per window and feature part (hand = 127 registered specs, "
    "GU/GI/GD/GA = generated groups) written through dataset.get_or_build_features with "
    "the part's spec list (cache key covers the spec list); base file per window holds "
    "candidates, labels y_org/y_all and the novel flag, joined after the features; "
    "columns load lazily with polars scan_parquet column selection"
)
R0_RULE = (
    f"Phase 2 inner protocol, {R0_KEY}: gbdt.run_protocol(inner) with the 127 hand features "
    "and the frozen params file, fit on train_266 + train_273, early stopping and ndcg@10 "
    "(y_org, slice all, n_pos = all window positives) on train_280; must equal the logged "
    f"inner value of the newest runs/*-exp-E1.json within {R0_TOL:g}"
)
TWIN_SOURCE = Path(__file__).resolve().parents[1] / "tests" / "test_features_gen.py"
HAND_TOP10 = (
    "ui_decay_org_hl14",
    "ui_orgpos_30",
    "ui_share_u_30",
    "rank_user",
    "ui_decay_all_hl7",
    "ui_rank_user_all",
    "u_novel_share_30",
    "ui_pos_30",
    "ui_pos_7",
    "ui_like_state",
)
HAND_TOP10_SOURCE = "docs/gbdt.md section 5, E1a seed 42 gain"
SCREEN_RULES = {
    "candidates": "generated features only; the 13 C/Z candidate-source and context "
    "features are fixed inputs of every arm and are not screened",
    "auc": f"pooled ROC-AUC against {af.SCREEN_LABEL} over all candidate rows of a window, "
    "average ranks for ties, missing values (null or NaN) rank below every value",
    "ndcg": f"single-feature ndcg@{af.SCREEN_K}, {af.SCREEN_LABEL}, per-user ranking by the "
    "oriented feature, ties by item_id ascending, n_pos = all window positives, users "
    "with n_pos >= 1 (users without candidates count as 0), as metrics.rank_metrics",
    "direction": "higher is better if the AUC pooled over the S1 fit windows "
    f"{list(af.SCREEN_FIT)} is > 0.5, otherwise lower is better; lower is better "
    "reverses the whole order, so missing values move to the top",
    "ranking": "mean oriented ndcg@10 over the screening windows, ties by mean oriented "
    "AUC over the same windows, then by name",
    "constant": "no missing value and a single value over the screening windows",
    "all_null": "no present value over the screening windows",
    "duplicate": f"sha256 over missing mask and values of {list(af.HASH_WINDOWS)}; the "
    "better ranked column is kept",
    "prune": f"greedy in rank order, drop if |Spearman| > {af.RHO_MAX} with a kept "
    f"feature, {af.RHO_SHARE:.0%} user sample (seed {af.SEED}) of rows of "
    f"{list(af.RHO_WINDOWS)} stacked, missing values rank lowest, stop at "
    f"{af.POOL_MAX} kept features",
}

SELECT_RULES = {
    "candidates": f"the {af.POOL_MAX}-feature screening pool (R2 pool, newest "
    "runs/*-autofeat-screen.json) is the selectable set; the 13 C/Z candidate-source and "
    "context features are fixed inputs of every fit and are never selected",
    "fits": f"R3: folds S1 (fit train_252 + train_259, early stopping and valid train_266) and "
    f"S2 (fit train_259 + train_266, early stopping train_273) x seeds {list(af.R3_SEEDS)}; "
    "frozen params of runs/gbdt-best-params.json, lambdarank, label y_org, early stopping "
    "patience 100 on the fold valid window with ndcg@10 (y_org, slice all, n_pos = all "
    "window positives), num_threads 8, deterministic; train_280, val and test are never loaded",
    "gain_share": "gain of a pool feature at the best iteration divided by the total gain of "
    "the pool features in the same fit (C/Z gain excluded from the denominator)",
    "clusters": f"connected components of the pool features at |Spearman| > {af.CLUSTER_RHO} "
    f"on the screening rho sample ({af.RHO_SHARE:.0%} of users, seed {af.SEED}, rows of "
    f"{list(af.RHO_WINDOWS)} stacked, missing values rank lowest)",
    "stability": "per fit, clusters are ranked by cluster gain share (sum of member shares, "
    "ties by cluster id = best member pool rank); a cluster counts in a fit if it is among "
    f"the first M clusters, M = number of whole clusters needed to cover {af.COVER_FACTOR}K "
    "features (the plan's top-2K applied to cluster units); a cluster is stable if it counts "
    f"in at least {af.STABLE_MIN_FITS} of 6 fits",
    "fill": "stable clusters in order of mean cluster share (over the 6 fits); inside a "
    "cluster members by mean member share, ties by pool rank; at most "
    f"{af.CLUSTER_CAP} members per cluster; stop at K; if fewer than K, fill the remainder "
    "with the highest mean share features of non-stable clusters (at most "
    f"{af.CLUSTER_CAP} per cluster); the number filled this way is logged",
    "jaccard": "per fit top-K pool features by gain share (ties by pool rank); pairwise "
    "Jaccard over the 6 fits, means within and across folds, and S1 vs S2 on the fold mean "
    "share top-K",
    "greedy": f"R4 family-wise forward selection; family = (entity, signal, agg), all windows "
    f"and half-lives together, pool members only; candidates = the top {af.GREEDY_FAMILIES} "
    "families by best member pool rank; base = C/Z; fold S2 (fit train_259 + train_266, "
    f"valid train_273), seed {af.GREEDY_SEEDS[0]}; each step adds the family with the best "
    "valid ndcg@10 gain (y_org, slice all) over the current set, ties by family rank; the "
    f"threshold is max({af.GREEDY_MIN_GAIN}, std) with std (ddof 1) of the proxy score of "
    f"base + the step-1 best family over seeds {list(af.GREEDY_SEEDS)}, measured once; stop "
    f"when the best gain < threshold (step 1 included), after {af.GREEDY_MAX_STEPS} steps, "
    "or when the wall-time budget is used (checked before every fit; the partial result "
    "keeps the completed steps and is flagged timed_out)",
    "proxy": f"greedy model = frozen params with learning_rate {af.PROXY_LEARNING_RATE}, "
    f"fixed {af.PROXY_ROUNDS} boosting rounds without early stopping, same objective, "
    "label, threads and determinism, scored once on the valid window after the last round",
    "r4_cut": "R4 list = pool members of the chosen families; if the union has more than K "
    "members, keep the K with the highest R3 mean member share (ties by pool rank); order = "
    "family order, inside a family by R3 mean share; K = 114 and K = 40 use the same "
    "ordered family list",
    "lists": "runs/autofeat-select-{R2,R3,R4}-K{114,40}.json with method, K, features in "
    "order, sha256 of the sorted list, sha256 of the ordered list and the producing run id; "
    "R2 lists are copied from the screening log; existing files are never overwritten, a "
    "rerun compares its lists with them",
}


def load_params(path: str | Path) -> tuple[dict, dict]:
    raw = Path(path).read_bytes()
    params = json.loads(raw)
    if isinstance(params, Mapping) and "selected_params" in params:
        params = params["selected_params"]
    params = dict(params)
    gb.GBDTConfig(params=params)
    return params, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def r0_reference(runs_dir: str | Path = "runs") -> dict | None:
    logs = sorted(Path(runs_dir).glob("*-exp-E1.json"))
    if not logs:
        return None
    body = json.loads(logs[-1].read_text())
    value = body["metrics"]["inner"]["train_280"]["y_org"]["all"][R0_KEY]["ndcg@10"]
    return {"log": str(logs[-1]), "key": R0_KEY, "value": value}


def dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def disk_state(cache_dir: Path) -> dict:
    target = cache_dir if cache_dir.exists() else cache_dir.parent
    return {
        "free_bytes": shutil.disk_usage(target).free,
        "cache_bytes": dir_bytes(cache_dir),
    }


class DiskSpaceError(RuntimeError):
    pass


def check_disk(free: int, projected: int, min_free_gb: float) -> None:
    left = (free - projected) / GIB
    if left < min_free_gb:
        raise DiskSpaceError(
            f"projected free space after the build is {left:.1f} GB "
            f"(free {free / GIB:.1f} GB, projected new cache {projected / GIB:.1f} GB), "
            f"below the {min_free_gb:g} GB floor"
        )


def window_config() -> dict:
    return {
        w.name: {
            "T": w.T,
            "T_end": w.T_end,
            "T_day": w.T / DAY,
            "label_end_day": w.T_end / DAY,
        }
        for w in af.autofeat_windows()
    }


def stage_build(args) -> int:
    start = time.perf_counter()
    runtime: dict[str, float] = {}
    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache
    folds = af.selection_folds()
    af.assert_selection_horizon(af.selection_window_names())
    disk0 = disk_state(cache_dir)
    print(
        f"disk: free {disk0['free_bytes'] / GIB:.1f} GB, "
        f"cache {disk0['cache_bytes'] / GIB:.2f} GB at {cache_dir}"
    )
    check_disk(disk0["free_bytes"], 0, args.min_free_gb)
    params, params_source = load_params(args.params_json)
    hand = af.hand_specs()
    generated = {g: af.generated_specs(g) for g in af.GENERATED_GROUPS}
    t0 = time.perf_counter()
    uids = ds.sample_users(args.data, args.users)
    events, info = af.load_selection_events(args.data, uids)
    runtime["load"] = time.perf_counter() - t0
    print(
        f"loaded {info['loaded_rows']} rows, {info['loaded_users']} users, "
        f"max timestamp {info['loaded_max_ts']} < val start {af.LOAD_LIMIT}"
    )

    def report(w, part, res):
        print(
            f"  {w.name} {part:4} specs {res['specs']:4} "
            f"{'hit ' if res['hit'] else 'miss'} {res['seconds']:7.1f} s "
            f"{res['bytes'] / 1024**2:8.1f} MB peak_rss {gb.peak_rss_mb():.0f} MB",
            flush=True,
        )

    builds, cache_hits, nbytes, peak = [], {}, {}, {}
    names = [w.name for w in af.autofeat_windows()]
    for i, name in enumerate(names):
        before = dir_bytes(cache_dir)
        b = af.build_window(
            events, name, DEFAULT_CONFIG, hand, generated, use_cache, cache_dir, report
        )
        builds.append(b)
        added = dir_bytes(cache_dir) - before
        runtime.update({f"{k}/{name}": v for k, v in b["timings"].items()})
        cache_hits[name] = {
            **b["cache"],
            **{p: r["hit"] for p, r in b["parts"].items()},
        }
        nbytes[name] = {p: r["bytes"] for p, r in b["parts"].items()}
        nbytes[name]["generated_total"] = sum(
            r["bytes"] for p, r in b["parts"].items() if p != af.HAND
        )
        nbytes[name]["new_in_cache"] = added
        peak[name] = gb.peak_rss_mb()
        c = b["counts"]
        print(
            f"{name}: rows {c['rows']} users {c['users']} pos_y_org {c['pos_y_org']} "
            f"generated {nbytes[name]['generated_total'] / GIB:.2f} GB "
            f"peak_rss {peak[name]:.0f} MB",
            flush=True,
        )
        if i == 0 and added > 0:
            projected = added * (len(names) - 1)
            free = shutil.disk_usage(cache_dir).free
            print(
                f"disk projection: {added / GIB:.2f} GB per window, "
                f"{projected / GIB:.2f} GB more, free {free / GIB:.1f} GB"
            )
            check_disk(free, projected, args.min_free_gb)
    manifest = af.write_manifest(builds, args.users, cache_dir)
    del events
    t0 = time.perf_counter()
    r0 = af.reproduce_r0(params, args.users, cache_dir, gb.SEED, runtime)
    runtime["r0/total"] = time.perf_counter() - t0
    reference = r0_reference(args.runs_dir) if args.users == 0 else None
    value = r0["value"]
    if reference is None:
        check = {
            "expected": None,
            "delta": None,
            "match": None,
            "note": "the logged reference exists only for the full dev sample",
        }
    else:
        delta = value - reference["value"]
        check = {
            "expected": reference["value"],
            "expected_source": reference["log"],
            "delta": delta,
            "match": abs(delta) <= R0_TOL,
        }
    catalog = af.feature_catalog()
    record = {
        "seed": gb.SEED,
        "seeds": [gb.SEED],
        "git_commit": gb.GIT_COMMIT,
        "stage": "build",
        "protocol": "inner",
        "selection_window": r0["fold"]["valid"],
        "selection_windows": list(af.selection_window_names()),
        "config": {
            "candidates": DEFAULT_CONFIG.config_dict(),
            "windows": window_config(),
            "folds": [f._asdict() for f in folds],
            "selection_limit_day": af.SELECTION_LIMIT_DAY,
            "load_limit": af.LOAD_LIMIT,
            "users_sample": {
                "n": args.users,
                "seed": gb.SEED,
                "pool": "uids with timestamp < VAL_TS",
            },
            "use_cache": use_cache,
            "storage": STORAGE,
            "params": params,
            "params_source": params_source,
            "r0_rule": R0_RULE,
            "r0_tolerance": R0_TOL,
            "min_free_gb": args.min_free_gb,
        },
        "features": {
            "hand": len(hand),
            "generated": {g: len(s) for g, s in generated.items()},
            "generated_total": sum(len(s) for s in generated.values()),
            "catalog_rows": catalog.height,
            "spec_count": gen.spec_count(af.generated_specs()),
            "feature_version": ft.FEATURE_VERSION,
            "source_sha256": fc.source_digest(),
        },
        "data": info,
        "versions": gb.library_versions(),
        "windows": {
            b["window"].name: {"counts": b["counts"], "files": b["files"]}
            for b in builds
        },
        "manifest": manifest.name,
        "r0": {
            "value": value,
            "value_rounded": round(value, bl.DIGITS),
            **check,
            "fold": r0["fold"],
            "n_features": r0["n_features"],
            "inner": r0["inner"],
            "gbdt": r0["config"],
        },
        "metrics": {"r0": {"train_280": {"y_org": {"all": {"ndcg@10": value}}}}},
    }
    runtime["total"] = time.perf_counter() - start
    record["runtime_seconds"] = runtime
    record["resources"] = {
        "peak_rss_mb": gb.peak_rss_mb(),
        "peak_rss_mb_after_window": peak,
        "cache_hits": cache_hits,
        "parquet_bytes": nbytes,
        "disk_before": disk0,
        "disk_after": disk_state(cache_dir),
    }
    path, digest = gb.write_run_log(record, args.name, Path(args.out_dir))
    gen_total = sum(n["generated_total"] for n in nbytes.values())
    print(f"generated cache total {gen_total / GIB:.2f} GB over {len(builds)} windows")
    print(
        f"R0 ndcg@10 (y_org, all, train_280, seed {gb.SEED}): {value:.6f}"
        + (
            ""
            if check["expected"] is None
            else f" expected {check['expected']:.6f} delta {check['delta']:+.2e} "
            f"match {check['match']}"
        )
    )
    print(f"best_iteration {r0['inner']['best_iteration']}")
    print("timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in runtime.items()))
    print(f"peak_rss_mb {record['resources']['peak_rss_mb']:.0f}")
    print(f"run log: {path}")
    print(f"metrics_digest: {digest}")
    if check["match"] is False:
        print(
            f"R0 MISMATCH: {value:.6f} vs logged {check['expected']:.6f}",
            file=sys.stderr,
        )
        return R0_EXIT_CODE
    return 0


def hand_twins(path: str | Path = TWIN_SOURCE) -> dict[str, dict[str, str]]:
    path = Path(path)
    found: dict[str, dict[str, str]] = {}
    if path.is_file():
        for node in ast.parse(path.read_text()).body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("TWINS", "CLOSE_TWINS")
            ):
                found[node.targets[0].id] = dict(ast.literal_eval(node.value))
    return {"exact": found.get("TWINS", {}), "close": found.get("CLOSE_TWINS", {})}


def composition(names: Sequence[str], catalog: pl.DataFrame) -> dict:
    meta = {r["name"]: r for r in catalog.iter_rows(named=True)}
    out = {}
    for key in ("group", "entity", "signal", "agg", "window"):
        values = [meta[n][key] for n in names]
        if key == "window":
            values = ["all" if v is None else f"w{v}" for v in values]
        counts = Counter(values)
        out[key] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))))
    return out


def _holder(rows: Mapping[str, Mapping], name: str) -> str | None:
    seen = set()
    while name is not None and name not in seen:
        seen.add(name)
        row = rows[name]
        if row["status"] == "kept":
            return name
        name = row["representative"]
    return None


def twin_overlap(
    selected: pl.DataFrame, lists: Mapping[str, Sequence[str]], twins: Mapping
) -> dict:
    rows = {r["name"]: r for r in selected.iter_rows(named=True)}
    sets = {k: set(v) for k, v in lists.items()}
    pairs = {**twins["exact"], **twins["close"]}

    def where(gname):
        if gname not in rows:
            return None
        r = rows[gname]
        holder = _holder(rows, gname)
        return {
            "generated": gname,
            "status": r["status"],
            "rank": r["rank"],
            "pool_rank": r["pool_rank"],
            "representative": r["representative"],
            "rho": r["rho"],
            "holder": holder,
            **{f"in_{k}": gname in s for k, s in sets.items()},
            **{f"covered_{k}": holder in s for k, s in sets.items()},
        }

    found = {g: where(g) for g in pairs if g in rows}
    summary = {
        "twins_total": len(pairs),
        "twins_screened": len(found),
        **{
            f"twins_in_{k}": sum(v[f"in_{k}"] for v in found.values()) for k in sets
        },
        **{
            f"twins_covered_{k}": sum(v[f"covered_{k}"] for v in found.values())
            for k in sets
        },
    }
    inverse = {h: g for g, h in pairs.items()}
    top10 = {
        h: {"twin": inverse.get(h), **(found.get(inverse.get(h)) or {})}
        for h in HAND_TOP10
    }
    for k in sets:
        summary[f"top10_hand_covered_{k}"] = sum(
            bool(v.get(f"covered_{k}")) for v in top10.values()
        )
    return {
        "source": str(TWIN_SOURCE.relative_to(TWIN_SOURCE.parents[1])),
        "rule": "exact and close twins from tests/test_features_gen.py (bit-equal or "
        "near-equal on the fixture); a twin is covered by a list when it, or the kept "
        "feature it was pruned or deduplicated to, is in the list; screening windows only, "
        "the H3 check on train_280 is not part of this stage",
        "hand_top10_source": HAND_TOP10_SOURCE,
        "summary": summary,
        "hand_top10": top10,
        "twins": {pairs[g]: v for g, v in sorted(found.items())},
    }


def stage_screen(args) -> int:
    start = time.perf_counter()
    runtime: dict[str, float] = {}
    cache_dir = Path(args.cache_dir)
    windows = af.check_screen_windows()
    if windows != af.selection_window_names():
        raise RuntimeError(f"screening windows {windows} are not the selection windows")
    man = af.read_manifest(args.users, cache_dir)
    present = {c for g in af.GENERATED_GROUPS for c in man["columns"].get(g, [])}
    features = [s.name for s in af.generated_specs() if s.name in present]
    loaded: list[str] = []

    def load(window, columns):
        loaded.append(window)
        return af.screen_frame(window, columns, args.users, cache_dir)

    def npos(window):
        loaded.append(window)
        return af.screen_npos(window, args.users, cache_dir)

    def progress(batch, done, total):
        print(
            f"  batch {batch:3d} {done:5d}/{total} features "
            f"{time.perf_counter() - start:7.1f} s peak_rss {gb.peak_rss_mb():.0f} MB",
            flush=True,
        )

    print(f"screening {len(features)} generated features on {list(windows)}")
    table, z, info = af.screen_features(
        features,
        load,
        npos,
        windows,
        batch_size=args.batch_size,
        workers=args.workers,
        timings=runtime,
        on_batch=progress,
    )
    used = [w for w in af.WINDOW_NAMES if w in set(loaded)]
    if tuple(used) != windows or "train_280" in loaded:
        raise RuntimeError(f"screening read windows {used}, expected {list(windows)}")
    t0 = time.perf_counter()
    selected = af.select_pool(table, z)
    runtime["screen/prune"] = time.perf_counter() - t0
    del z
    catalog = af.feature_catalog()
    selected = selected.join(
        catalog.select("name", "group", "entity", "signal", "agg", "window"),
        on="name",
        how="left",
    )
    lists = af.pool_lists(selected)
    counts = {"candidates": selected.height}
    counts.update({s: int((selected["status"] == s).sum()) for s in af.SCREEN_STATUSES})
    counts["after_constant_null"] = counts["candidates"] - counts["constant"]
    counts["after_constant_null"] -= counts["all_null"]
    counts["after_dedup"] = counts["after_constant_null"] - counts["duplicate"]
    counts["pool"] = len(lists["pool"])
    pruned_map = {
        r["name"]: {"status": r["status"], "representative": r["representative"], "rho": r["rho"]}
        for r in selected.filter(pl.col("status").is_in(["duplicate", "pruned"]))
        .sort("rank")
        .iter_rows(named=True)
    }
    stat_cols = [
        "rank",
        "pool_rank",
        "name",
        "direction",
        "ndcg_mean",
        "oauc_mean",
        "oauc_fit",
        *(f"ndcg_{w}" for w in windows),
        *(f"oauc_{w}" for w in windows),
        "missing_share",
    ]
    top15 = (
        selected.filter(pl.col("status") == "kept")
        .sort("pool_rank")
        .head(15)
        .select(stat_cols)
        .to_dicts()
    )
    kept = selected.filter(pl.col("status") == "kept")
    summary = {}
    for key, names in lists.items():
        sub = kept.filter(pl.col("name").is_in(names))
        summary[key] = {
            "n": len(names),
            "ndcg_mean_avg": float(sub["ndcg_mean"].mean()) if names else None,
            "oauc_fit_avg": float(sub["oauc_fit"].mean()) if names else None,
        }
    twins = twin_overlap(selected, lists, hand_twins())
    now = datetime.now()
    run_id = f"{now:%Y%m%d-%H%M%S}-{args.name}"
    table_path = Path(args.out_dir) / f"{run_id}-table.parquet"
    for p in (table_path, Path(args.out_dir) / f"{run_id}.json"):
        if p.exists():
            raise FileExistsError(f"{p} already exists")
    ordered = ["rank", "pool_rank", "status", "name", "group", "entity", "signal", "agg"]
    ordered += ["window", "representative", "rho"]
    ds._write_atomic(
        selected.select(*ordered, pl.exclude(ordered)).sort("rank"), table_path
    )
    record = {
        "seed": af.SEED,
        "seeds": [af.SEED],
        "git_commit": gb.GIT_COMMIT,
        "stage": "screen",
        "protocol": "inner",
        "selection_window": windows[-1],
        "selection_windows": list(windows),
        "loaded_windows": used,
        "config": {
            "version": af.SCREEN_VERSION,
            "label": af.SCREEN_LABEL,
            "slice": af.SCREEN_SLICE,
            "k": af.SCREEN_K,
            "windows": list(windows),
            "fit_windows": list(af.SCREEN_FIT),
            "hash_windows": list(af.HASH_WINDOWS),
            "rho_windows": list(af.RHO_WINDOWS),
            "rho_max": af.RHO_MAX,
            "rho_share": af.RHO_SHARE,
            "rho_block": af.RHO_BLOCK,
            "pool_max": af.POOL_MAX,
            "top_ks": list(af.TOP_KS),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "users_sample": {"n": args.users, "seed": af.SEED},
            "folds": [f._asdict() for f in af.selection_folds()],
            "selection_limit_day": af.SELECTION_LIMIT_DAY,
            "rules": SCREEN_RULES,
        },
        "features": {
            "screened": len(features),
            "manifest_generated": len(present),
            "feature_version": ft.FEATURE_VERSION,
            "source_sha256": fc.source_digest(),
        },
        "data": info,
        "versions": gb.library_versions(),
        "counts": counts,
        "lists": lists,
        "sha256": {
            k: {"sorted": af.list_sha256(v), "ranked": af.list_sha256(v, True)}
            for k, v in lists.items()
        },
        "top15": top15,
        "composition": {k: composition(v, catalog) for k, v in lists.items()},
        "hand_twins": twins,
        "pruned_map": pruned_map,
        "metrics": {"screen": summary},
    }
    runtime["total"] = time.perf_counter() - start
    record["runtime_seconds"] = runtime
    record["resources"] = {"peak_rss_mb": gb.peak_rss_mb(), "table_file": table_path.name}
    path, digest = gb.write_run_log(record, args.name, Path(args.out_dir), now)
    print(f"selection_windows {record['selection_windows']} loaded {used}")
    print("counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("top 15 (pool_rank name dir ndcg_mean oauc_mean oauc_fit):")
    for r in top15:
        print(
            f"  {r['pool_rank']:3d} {r['name']:40} {r['direction']:+d} "
            f"{r['ndcg_mean']:.4f} {r['oauc_mean']:.4f} {r['oauc_fit']:.4f}"
        )
    for key in ("top114", "top40"):
        comp = record["composition"][key]
        print(f"{key}: " + "; ".join(f"{k} {v}" for k, v in comp.items() if k != "window"))
    print("hand twins: " + json.dumps(twins["summary"]))
    for k, v in record["sha256"].items():
        print(f"sha256 {k}: {v['sorted']}")
    print("timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in runtime.items()))
    print(f"peak_rss_mb {record['resources']['peak_rss_mb']:.0f}")
    print(f"table: {table_path}")
    print(f"run log: {path}")
    print(f"metrics_digest: {digest}")
    return 0


def newest_log(runs_dir: str | Path, pattern: str, accept=None) -> Path:
    for path in sorted(Path(runs_dir).glob(pattern), reverse=True):
        if accept is None or accept(json.loads(path.read_text())):
            return path
    raise FileNotFoundError(f"no {pattern} log in {runs_dir}")


def write_list_file(path: Path, body: Mapping) -> str:
    if path.exists():
        old = json.loads(path.read_text())
        same = old.get("features") == body["features"] and old.get("sha256") == body["sha256"]
        return "identical" if same else "conflict"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as fh:
        fh.write(json.dumps(body, indent=2) + "\n")
    return "written"


def cluster_stats(clusters: Sequence[Sequence[int]], corr, pool: Sequence[str]) -> dict:
    sizes = [len(m) for m in clusters]
    off = np.abs(corr - np.diag(np.diag(corr)))
    big = sorted(range(len(clusters)), key=lambda c: (-sizes[c], c))[:10]
    return {
        "n_features": len(pool),
        "n_clusters": len(clusters),
        "n_singletons": sum(s == 1 for s in sizes),
        "max_size": max(sizes),
        "size_counts": {str(k): v for k, v in sorted(Counter(sizes).items())},
        "max_abs_rho_pool": float(off.max()) if off.size else None,
        "largest": [
            {"cluster": c, "size": sizes[c], "members": [pool[j] for j in clusters[c]]}
            for c in big
        ],
        "members": {pool[m[0]]: [pool[j] for j in m] for m in clusters if len(m) > 1},
    }


def stage_select(args) -> int:
    start = time.perf_counter()
    runtime: dict[str, float] = {}
    cache_dir = Path(args.cache_dir)
    folds = af.select_folds()
    windows = af.select_window_names(folds)
    if windows != af.selection_window_names():
        raise RuntimeError(f"select windows {windows} are not the selection windows")
    screen_path = (
        Path(args.screen_log)
        if args.screen_log
        else newest_log(args.runs_dir, "*-autofeat-screen.json")
    )
    screen = json.loads(screen_path.read_text())
    pool = list(screen["lists"]["pool"])
    if af.list_sha256(pool, True) != screen["sha256"]["pool"]["ranked"]:
        raise RuntimeError(f"pool list of {screen_path} does not match its sha256")
    table = pl.read_parquet(screen_path.parent / screen["resources"]["table_file"])
    fixed = list(af.fixed_features())
    columns = [*fixed, *pool]
    params, params_source = load_params(args.params_json)
    catalog = af.feature_catalog()
    families = af.pool_families(pool, catalog)
    candidates = dict(list(families.items())[: af.GREEDY_FAMILIES])
    pool_rank = {n: i + 1 for i, n in enumerate(pool)}
    print(f"pool {len(pool)} features from {screen_path.name}, fixed {len(fixed)} C/Z")
    for key in ("stability", "fill"):
        print(f"rule {key}: {SELECT_RULES[key]}")
    loaded: list[str] = []
    datas: dict[str, ds.WindowData] = {}

    def need(names):
        for n in names:
            if n not in datas:
                t = time.perf_counter()
                datas[n] = af.select_window_data(n, columns, args.users, cache_dir)
                loaded.append(n)
                runtime[f"load/{n}"] = time.perf_counter() - t
                print(
                    f"  loaded {n} {datas[n].frame.shape} peak_rss {gb.peak_rss_mb():.0f} MB",
                    flush=True,
                )

    def release(keep):
        for n in list(datas):
            if n not in keep:
                del datas[n]

    need(af.RHO_WINDOWS)
    t0 = time.perf_counter()
    frames = {
        w: datas[w].frame.select(*ds.PAIR, af.SCREEN_LABEL, *pool) for w in af.RHO_WINDOWS
    }
    npos = {
        w: datas[w]
        .n_pos[af.SCREEN_LABEL]
        .select("uid", pl.col(af.SCREEN_SLICE).alias("n_pos"))
        for w in af.RHO_WINDOWS
    }
    z, zinfo = af.cluster_z(frames, npos, pool)
    del frames
    corr = af.rho_matrix(z)
    del z
    clusters = af.rho_clusters(corr, af.CLUSTER_RHO)
    cstats = {**cluster_stats(clusters, corr, pool), "sample": zinfo}
    del corr
    runtime["clusters"] = time.perf_counter() - t0
    print(
        f"clusters: {cstats['n_clusters']} ({cstats['n_singletons']} singletons, max size "
        f"{cstats['max_size']}), max |rho| in pool {cstats['max_abs_rho_pool']:.4f}",
        flush=True,
    )
    fits, shares, fit_seconds = [], [], {}
    for fold in folds:
        release({*fold.train, fold.valid})
        need((*fold.train, fold.valid))
        cfg0 = af.select_config(params, af.R3_SEEDS[0], fold.valid)
        train, valid, n_pos, dropped = af.fold_frames(datas, fold, columns, cfg0)
        for seed in af.R3_SEEDS:
            cfg = af.select_config(params, seed, fold.valid)
            t = time.perf_counter()
            res = gb.fit(train, valid, cfg, n_pos, columns)
            name = f"{fold.name}:s{seed}"
            fit_seconds[name] = time.perf_counter() - t
            gains = gb.gain_importance(res)["by_feature"]
            share = af.gain_shares(gains, pool)
            total = float(sum(gains.values()))
            gain_fixed = float(sum(gains[n] for n in fixed))
            fits.append(
                {
                    "fit": name,
                    "fold": fold.name,
                    "seed": seed,
                    "train_windows": list(fold.train),
                    "valid_window": fold.valid,
                    "best_iteration": res.best_iteration,
                    "best_ndcg@10_in_selection": res.best_score,
                    "rounds_run": len(res.curve),
                    "train_rows": res.train_rows,
                    "train_groups": res.train_groups,
                    "dropped_zero_positive_groups": dropped,
                    "gain_total": total,
                    "gain_fixed": gain_fixed,
                    "gain_pool": total - gain_fixed,
                    "fixed_share_of_total": gain_fixed / total,
                    "pool_features_with_gain": int((share > 0).sum()),
                }
            )
            shares.append(share)
            print(
                f"  fit {name}: best_iter {res.best_iteration} ndcg@10 {res.best_score:.6f} "
                f"(in-selection) C/Z share {gain_fixed / total:.3f} "
                f"{fit_seconds[name]:.0f} s peak_rss {gb.peak_rss_mb():.0f} MB",
                flush=True,
            )
            del res
        del train, valid
    runtime["r3/fits"] = sum(fit_seconds.values())
    share_arr = np.stack(shares)
    fit_names = [f["fit"] for f in fits]
    fold_of = {f["fit"]: f["fold"] for f in fits}
    mean = share_arr.mean(axis=0)
    mean_share = dict(zip(pool, mean.tolist()))
    labels = af.cluster_labels(clusters, len(pool))
    gain_share = {
        n: {
            "mean": float(mean[j]),
            "std": float(share_arr[:, j].std(ddof=1)),
            "cluster": int(labels[j]),
            "pool_rank": j + 1,
            "shares": share_arr[:, j].tolist(),
        }
        for j, n in enumerate(pool)
    }
    by_mean = sorted(range(len(pool)), key=lambda j: (-mean[j], j))
    r3_sel, jac, lists = {}, {}, {}
    for k in af.TOP_KS:
        res = af.stability_select(share_arr, clusters, k)
        alt = af.stability_select(share_arr, clusters, k, cover=k)
        names = [pool[j] for j in res["selected"]]
        lists[f"R3_K{k}"] = names
        r3_sel[f"K{k}"] = {
            "k": k,
            "cover": res["cover"],
            "m_per_fit": dict(zip(fit_names, res["m_per_fit"])),
            "n_stable_clusters": res["n_stable_clusters"],
            "n_stable_features_available": int(
                sum(len(clusters[c]) for c in res["stable"])
            ),
            "n_from_stable": res["n_from_stable"],
            "n_filled_from_non_stable": res["n_filled"],
            "n_selected": len(names),
            "stable_clusters": [
                {
                    "cluster": c,
                    "hits": res["hits"][c],
                    "mean_share": res["cluster_mean_share"][c],
                    "members": [pool[j] for j in clusters[c]],
                }
                for c in res["stable"]
            ],
            "hits_histogram": {
                str(h): v for h, v in sorted(Counter(res["hits"]).items())
            },
            "diagnostic_cover_K": {
                "note": "sensitivity only, not used for selection",
                "n_stable_clusters": alt["n_stable_clusters"],
                "n_filled_from_non_stable": alt["n_filled"],
            },
        }
        tops = dict(zip(fit_names, af.topk_indices(share_arr, k)))
        fold_tops = {
            f.name: af.topk_indices(
                share_arr[[i for i, n in enumerate(fit_names) if fold_of[n] == f.name]].mean(
                    axis=0
                ),
                k,
            )[0]
            for f in folds
        }
        jac[f"K{k}"] = {
            **af.jaccard_summary(tops, fold_of),
            "fold_mean_share_S1_vs_S2": af.jaccard(fold_tops["S1"], fold_tops["S2"]),
            "r3_selected_vs_mean_share_topk": af.jaccard(res["selected"], by_mean[:k]),
        }
    top15 = [
        {"name": pool[j], "mean_share": float(mean[j]), "pool_rank": j + 1, "cluster": int(labels[j])}
        for j in by_mean[:15]
    ]
    fit_sha = {
        "shares_sha256": hashlib.sha256(share_arr.astype("<f8").tobytes()).hexdigest()
    }
    t0 = time.perf_counter()
    greedy_timings = None
    if args.skip_greedy:
        prev_path = (
            Path(args.greedy_from)
            if args.greedy_from
            else newest_log(
                args.runs_dir,
                "*-autofeat-select.json",
                lambda b: not b.get("greedy", {}).get("skipped", True),
            )
        )
        prev = json.loads(prev_path.read_text())
        if prev["greedy"].get("skipped", True):
            raise RuntimeError(f"{prev_path} holds no greedy trace")
        if prev["greedy"]["candidates"] != {k: list(v) for k, v in candidates.items()}:
            raise RuntimeError(f"greedy candidates of {prev_path} differ from this pool")
        greedy = {**prev["greedy"], "skipped": True, "source_run_id": prev["run_id"]}
        print(f"greedy skipped; trace read from {prev_path.name}")
    else:
        fold = next(f for f in folds if f.name == af.GREEDY_FOLD)
        release({*fold.train, fold.valid})
        need((*fold.train, fold.valid))
        gcols = [*fixed, *(n for m in candidates.values() for n in m)]
        cfg0 = af.select_config(params, af.GREEDY_SEEDS[0], fold.valid, proxy=True)
        train, valid, n_pos, dropped = af.fold_frames(datas, fold, gcols, cfg0)
        release(set())
        score = af.proxy_scorer(train, valid, n_pos, params, fold.valid)
        print(
            f"greedy: {len(candidates)} candidate families, {len(gcols) - len(fixed)} "
            f"members, budget {args.greedy_budget:.0f} s",
            flush=True,
        )

        def on_eval(step, fam, s, gain):
            print(
                f"  step {step:2d} {fam:32} ndcg@10 {s:.6f} gain {gain:+.6f} "
                f"{time.perf_counter() - t0:7.1f} s",
                flush=True,
            )

        trace = af.greedy_forward(
            candidates, fixed, score, budget=args.greedy_budget, on_eval=on_eval
        )
        greedy_timings = trace.pop("timings")
        del train, valid
        greedy = {
            "skipped": False,
            "fold": fold._asdict(),
            "proxy_config": cfg0.config_dict(),
            "train_dropped_zero_positive_groups": dropped,
            "candidates": {k: list(v) for k, v in candidates.items()},
            **trace,
        }
    release(set())
    runtime["greedy"] = time.perf_counter() - t0
    used = [w for w in af.WINDOW_NAMES if w in set(loaded)]
    if tuple(used) != windows or "train_280" in loaded:
        raise RuntimeError(f"select read windows {used}, expected {list(windows)}")
    r4 = {}
    for k in af.TOP_KS:
        out = af.family_list(greedy["chosen"], candidates, mean_share, pool_rank, k)
        lists[f"R4_K{k}"] = out["features"]
        r4[f"K{k}"] = {"union_size": out["union_size"], "cut": out["cut"], "n": len(out["features"])}
    for k in af.TOP_KS:
        lists[f"R2_K{k}"] = list(screen["lists"][f"top{k}"])
    now = datetime.now()
    run_id = f"{now:%Y%m%d-%H%M%S}-{args.name}"
    if (Path(args.out_dir) / f"{run_id}.json").exists():
        raise FileExistsError(f"{run_id}.json already exists")
    order = [f"{m}_K{k}" for m in af.SELECT_METHODS for k in af.TOP_KS]
    list_files, records = {}, {}
    for key in order:
        method, k = key.split("_K")
        source = screen["run_id"] if method == "R2" else run_id
        records[key] = af.list_record(method, int(k), lists[key], source)
        path = Path(args.out_dir) / f"autofeat-select-{method}-K{k}.json"
        list_files[key] = {"file": path.name, "status": write_list_file(path, records[key])}
    sets = {m: set(lists[f"{m}_K114"]) for m in af.SELECT_METHODS}
    overlap = {
        f"{a}&{b}": {"n": len(sets[a] & sets[b]), "jaccard": af.jaccard(sets[a], sets[b])}
        for i, a in enumerate(af.SELECT_METHODS)
        for b in af.SELECT_METHODS[i + 1 :]
    }
    overlap["R2&R3&R4"] = {"n": len(sets["R2"] & sets["R3"] & sets["R4"])}
    twins = twin_overlap(table, {k: lists[k] for k in order}, hand_twins())
    record = {
        "seed": af.SEED,
        "seeds": list(af.R3_SEEDS),
        "git_commit": gb.GIT_COMMIT,
        "stage": "select",
        "protocol": "inner",
        "selection_window": folds[-1].valid,
        "selection_windows": list(windows),
        "loaded_windows": used,
        "config": {
            "version": af.SELECT_VERSION,
            "label": af.SELECT_LABEL,
            "folds": [f._asdict() for f in folds],
            "select_limit_day": af.SELECT_LIMIT_DAY,
            "params": params,
            "params_source": params_source,
            "fit_config": af.select_config(params, af.SEED, folds[0].valid).config_dict(),
            "r3_seeds": list(af.R3_SEEDS),
            "cluster_rho": af.CLUSTER_RHO,
            "cover_factor": af.COVER_FACTOR,
            "stable_min_fits": af.STABLE_MIN_FITS,
            "cluster_cap": af.CLUSTER_CAP,
            "top_ks": list(af.TOP_KS),
            "greedy": {
                "fold": af.GREEDY_FOLD,
                "families": af.GREEDY_FAMILIES,
                "max_steps": af.GREEDY_MAX_STEPS,
                "min_gain": af.GREEDY_MIN_GAIN,
                "seeds": list(af.GREEDY_SEEDS),
                "budget_seconds": args.greedy_budget,
                "proxy_learning_rate": af.PROXY_LEARNING_RATE,
                "proxy_rounds": af.PROXY_ROUNDS,
                "skip_greedy": args.skip_greedy,
            },
            "screen_log": {
                "file": screen_path.name,
                "run_id": screen["run_id"],
                "pool_sha256": screen["sha256"]["pool"],
            },
            "users_sample": {"n": args.users, "seed": af.SEED},
            "rules": SELECT_RULES,
        },
        "features": {
            "pool": len(pool),
            "fixed": fixed,
            "columns": len(columns),
            "families_in_pool": len(families),
            "feature_version": ft.FEATURE_VERSION,
            "source_sha256": fc.source_digest(),
        },
        "versions": gb.library_versions(),
        "clusters": cstats,
        "r3": {
            "fits": fits,
            "fit_order": fit_names,
            **fit_sha,
            "top15": top15,
            "gain_share": gain_share,
            "selection": r3_sel,
            "jaccard": jac,
        },
        "greedy": greedy,
        "r4": r4,
        "lists": {k: lists[k] for k in order},
        "sha256": {
            k: {"sorted": records[k]["sha256"], "ranked": records[k]["sha256_ranked"]}
            for k in order
        },
        "composition": {k: composition(lists[k], catalog) for k in order},
        "overlap_K114": overlap,
        "hand_twins": twins,
        "metrics": {
            "r3_fits_in_selection": {
                f["fit"]: {f["valid_window"]: {"y_org": {"all": {"ndcg@10": f["best_ndcg@10_in_selection"]}}}}
                for f in fits
            },
            "greedy_proxy_s2": {
                "base": greedy["base_score"],
                "final": greedy["final_score"],
                "std": greedy["std"],
            },
        },
    }
    runtime["total"] = time.perf_counter() - start
    record["runtime_seconds"] = runtime
    record["resources"] = {
        "peak_rss_mb": gb.peak_rss_mb(),
        "fit_seconds": fit_seconds,
        "greedy_timings": greedy_timings,
        "list_files": list_files,
    }
    path, digest = gb.write_run_log(record, args.name, Path(args.out_dir), now)
    print(f"selection_windows {record['selection_windows']} loaded {used}")
    for k in af.TOP_KS:
        r = r3_sel[f"K{k}"]
        j = jac[f"K{k}"]
        print(
            f"R3 K{k}: stable clusters {r['n_stable_clusters']}, from stable "
            f"{r['n_from_stable']}, filled {r['n_filled_from_non_stable']}; jaccard all "
            f"{j['mean_all_pairs']:.3f} within S1 {j['mean_within_S1']:.3f} within S2 "
            f"{j['mean_within_S2']:.3f} cross {j['mean_cross_fold']:.3f} fold S1/S2 "
            f"{j['fold_mean_share_S1_vs_S2']:.3f}"
        )
    print("R3 top 15 by mean gain share:")
    for r in top15:
        print(f"  {r['pool_rank']:3d} {r['name']:40} {r['mean_share']:.4f} c{r['cluster']}")
    print(
        f"greedy: base {greedy['base_score']:.6f} final {greedy['final_score']:.6f} "
        f"std {greedy['std']} threshold {greedy['threshold']} stop {greedy['stop_reason']}"
    )
    for s in greedy["steps"]:
        if s.get("complete"):
            print(
                f"  step {s['step']}: {s['best_family']} gain {s['best_gain']:+.6f} "
                f"accepted {s['accepted']}"
            )
        else:
            print(f"  step {s['step']}: incomplete ({s['evaluated']} evaluated)")
    for key in order:
        comp = record["composition"][key]
        print(
            f"{key} n {len(lists[key])}: "
            + "; ".join(f"{k} {v}" for k, v in comp.items() if k in ("entity", "signal", "agg"))
        )
    print("overlap K114: " + json.dumps(overlap))
    print("R4 sizes: " + json.dumps(r4))
    print("hand twins: " + json.dumps(twins["summary"]))
    for key in order:
        print(
            f"sha256 {key}: {records[key]['sha256']} file {list_files[key]['file']} "
            f"{list_files[key]['status']}"
        )
    print("timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in runtime.items()))
    print(f"peak_rss_mb {record['resources']['peak_rss_mb']:.0f}")
    print(f"run log: {path}")
    print(f"metrics_digest: {digest}")
    if any(v["status"] == "conflict" for v in list_files.values()):
        print("LIST CONFLICT: an existing list file differs and was not overwritten", file=sys.stderr)
        return 1
    return 0


MAX_RSS_GB = 10.0
WATCH_INTERVAL = 1.0
ABORT_EXIT_CODE = 3
MIB = 1024**2
STEM_FORMAT = "%Y%m%d-%H%M%S"
BOOT_SLICES = ("all", "novel")
SUMMARY_METRICS = ("ndcg@10", "recall@10", "recall_cap@10", "hit_rate@10", "ndcg@100")
OVERLAP_BATCH = 150
R0_ARM = "R0"
R5_ARM = "R5_union"
R1_ARM = "R1_full"
R4_TINY = "R4_tiny"
FIRST_ARMS = (R0_ARM, "R2_K114", "R2_K40", "R3_K114", "R3_K40", R4_TINY, "R3b_K114", "R3b_K40")
REFERENCE_BASELINES = ("B3", "B5")
CONFIRM_RULES = {
    "fold": "S3: fit train_266 + train_273, early stopping and every metric on train_280; frozen "
    "params of runs/gbdt-best-params.json, lambdarank, label y_org, early stopping patience 100 "
    "by ndcg@10 (y_org, slice all, n_pos = all window positives), num_threads 8, deterministic",
    "data": "loaded windows are train_259 (near-copy sample only), train_266, train_273 and "
    "train_280; events for the baselines are read with timestamp < day 287; nothing after "
    "day 287 is read",
    "arms": "each arm = the listed generated features + the 13 fixed C/Z features; R0 = the 127 "
    "hand features (E1a); lists are read from runs/autofeat-select-*.json and are not changed",
    "metrics": "metrics.evaluate through baselines._evaluate_keys on train_280, probabilities "
    "for LogLoss from the percentile calibrator cross-fitted over user folds (as the Phase 2 "
    "inner protocol), all-users variants = metric * n_users / n_hist_users",
    "method_choice": "among R2 and R3 at K = 114: highest mean ndcg@10 (y_org, slice all, "
    "train_280) over seeds 42-44, per-seed values rounded to 6 decimals; if the difference to "
    "the other is within the larger seed std (ddof 1), the cheaper method wins (R2 before R3); "
    "R4 tiny, R3b and R5 are never eligible",
    "r3b": "POST-HOC: the R3 stability rule (same clusters, cover 2K, stable in at least 4 of "
    "6 fits, mean share order) without the cap of 3 members per cluster, from the gain shares "
    "of the six S1/S2 fits logged by the select stage; built before any S3 frame is loaded; "
    "reported on S3 only, not eligible for the method choice or the primary conclusion",
    "r4_tiny": "the 4-feature R4 list without padding, reported next to the others, not "
    "eligible at K = 114",
    "r5": "hand 127 + the winning K = 114 list minus near-copies of hand features: a generated "
    "feature is dropped when its largest |Spearman| to any hand feature is > 0.98 on the "
    "screening rho sample (20% of users, seed 42, rows of train_259 and train_266 stacked, "
    "missing values rank lowest); secondary",
    "r1": "all 1,360 generated + 13 C/Z features, seed 42 only, run last in a separate process "
    "under a 10 GB RSS watchdog; a watchdog trip is recorded as a finding without retry",
    "bootstrap": "paired user bootstrap on train_280, per-user ndcg@10 (y_org) averaged over the "
    "arm seeds, users with >= 1 positive of the slice, 1000 resamples, numpy default_rng(42) "
    "restarted for every comparison, percentile 95% CI",
    "overlap": "twins bit-equal on train_280 (exact twins) or rtol 1e-6 (close twin); max "
    "|Spearman| of each hand feature to the full grid and to the winning K = 114 list on a 20% "
    "user sample (seed 42) of train_280, missing values rank lowest; a hand group is recovered "
    "when at least one of its features has |rho| > 0.95",
    "h3": "at least 7 of the top-10 hand features (docs/gbdt.md section 5) have an exact twin or "
    "|Spearman| >= 0.95 to a grid feature on train_280, and G127 holds a pair decay feature "
    "with half-life 7, 14 or 30",
    "h4": "mean ndcg@10 of G127 minus mean ndcg@10 of G53 on train_280 <= 0.005",
    "h5": "generated pair-entity features carry >= 50% of the total gain (C/Z included in the "
    "denominator) of the G127 seed-42 fit",
}


def process_rss() -> int:
    return psutil.Process().memory_info().rss


class Watchdog(threading.Thread):
    def __init__(
        self,
        limit_bytes: float,
        state: dict,
        partial_path: Path,
        rss_fn: Callable[[], float] = process_rss,
        exit_fn: Callable[[int], object] = os._exit,
        interval: float = WATCH_INTERVAL,
    ):
        super().__init__(daemon=True)
        self.limit_bytes = limit_bytes
        self.state = state
        self.partial_path = Path(partial_path)
        self.rss_fn = rss_fn
        self.exit_fn = exit_fn
        self.interval = interval
        self.peak_bytes = 0.0
        self.started = time.time()
        self.stopper = threading.Event()

    def reset_peak(self) -> None:
        self.peak_bytes = float(self.rss_fn())

    def partial_record(self, rss: float) -> dict:
        return {
            "status": "aborted_rss_limit",
            "stage": self.state.get("stage"),
            "rss_bytes": int(rss),
            "rss_gb": rss / GIB,
            "limit_gb": self.limit_bytes / GIB,
            "seconds_since_start": time.time() - self.started,
            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
            "load_average": list(os.getloadavg()),
            "completed": list(self.state.get("completed", [])),
            "info": dict(self.state.get("info", {})),
        }

    def write_partial(self, rss: float) -> None:
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.partial_path.with_name(f".{self.partial_path.name}.tmp")
        tmp.write_text(json.dumps(bl._clean(self.partial_record(rss)), indent=2) + "\n")
        os.replace(tmp, self.partial_path)

    def step(self) -> bool:
        rss = self.rss_fn()
        self.peak_bytes = max(self.peak_bytes, rss)
        if rss <= self.limit_bytes:
            return False
        self.write_partial(rss)
        self.exit_fn(ABORT_EXIT_CODE)
        return True

    def run(self) -> None:
        while not self.stopper.wait(self.interval):
            if self.step():
                return

    def stop(self) -> None:
        self.stopper.set()


def project_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if resolved.is_relative_to(af.PROJECT_ROOT):
        return resolved.relative_to(af.PROJECT_ROOT).as_posix()
    return resolved.as_posix()


def read_lists(runs_dir: Path) -> tuple[dict[str, list[str]], dict[str, dict]]:
    lists, meta = {}, {}
    for m in af.SELECT_METHODS:
        for k in af.TOP_KS:
            path = runs_dir / f"autofeat-select-{m}-K{k}.json"
            body = json.loads(path.read_text())
            if af.list_sha256(body["features"]) != body["sha256"] or body["K"] != k:
                raise RuntimeError(f"{path} does not match its sha256 or K")
            key = f"{m}_K{k}"
            lists[key] = list(body["features"])
            meta[key] = {
                "file": project_path(path),
                "sha256": body["sha256"],
                "file_sha256": af.file_sha256(path),
                "run_id": body["run_id"],
                "n": len(body["features"]),
            }
    return lists, meta


def r3b_from_log(
    select: Mapping, screen: Mapping, lists: Mapping[str, Sequence[str]]
) -> tuple[dict[str, list[str]], dict]:
    pool = list(screen["lists"]["pool"])
    if af.list_sha256(pool, True) != select["config"]["screen_log"]["pool_sha256"]["ranked"]:
        raise RuntimeError("screening pool does not match the select log")
    gain_share = select["r3"]["gain_share"]
    if [n for n, _ in sorted(gain_share.items(), key=lambda kv: kv[1]["pool_rank"])] != pool:
        raise RuntimeError("logged gain shares are not in pool order")
    shares = af.shares_from_gain_table(gain_share, pool)
    clusters = af.clusters_from_members(pool, select["clusters"]["members"])
    for c, members in enumerate(clusters):
        for j in members:
            if gain_share[pool[j]]["cluster"] != c:
                raise RuntimeError("reconstructed clusters differ from the logged cluster ids")
    out, info = {}, {}
    for k in af.TOP_KS:
        capped = [pool[j] for j in af.stability_select(shares, clusters, k)["selected"]]
        if capped != list(lists[f"R3_K{k}"]):
            raise RuntimeError(f"R3 K{k} recomputed from the select log differs from its list file")
        res = af.r3b_select(shares, clusters, k)
        names = [pool[j] for j in res["selected"]]
        out[f"R3b_K{k}"] = names
        info[f"K{k}"] = {
            "n_stable_clusters": res["n_stable_clusters"],
            "n_from_stable": res["n_from_stable"],
            "n_filled_from_non_stable": res["n_filled"],
            "overlap_with_capped_R3": len(set(names) & set(lists[f"R3_K{k}"])),
            "jaccard_with_capped_R3": af.jaccard(names, lists[f"R3_K{k}"]),
            "capped_R3_reproduced_from_log": True,
        }
    info["source"] = {
        "select_log": select["run_id"],
        "shares_from": "r3.gain_share (six S1/S2 fits, rounded to 6 decimals in the log)",
        "clusters_from": "clusters.members (multi-member clusters) + singletons",
        "fits": select["r3"]["fit_order"],
    }
    return out, info


def summarize(metrics: Mapping, keys: Sequence[str]) -> dict:
    out = {}
    for lb, sl in (("y_org", "all"), ("y_org", "novel")):
        for m in SUMMARY_METRICS:
            vals = np.array([metrics[k][lb][sl][m] for k in keys], dtype=np.float64)
            out[f"{lb}/{sl}/{m}"] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else None,
            }
    return out


class ArmRunner:
    def __init__(self, datas, params, catalog, dog: Watchdog, state: dict):
        self.datas = datas
        self.params = params
        self.catalog = catalog
        self.dog = dog
        self.state = state
        self.sel = af.confirm_fold().valid
        self.metrics: dict[str, dict] = {}
        self.all_users: dict[str, dict] = {}
        self.vectors: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        self.uids: dict[str, np.ndarray] = {}
        self.checks: dict[str, float] = {}

    def add_vectors(self, arm: str, seed: int, scores, key: str) -> None:
        data = self.datas[self.sel]
        self.vectors.setdefault(arm, {})[seed] = {}
        for sl in BOOT_SLICES:
            u, v = af.per_user_ndcg(data.frame, scores, data.n_pos[af.CONFIRM_LABEL], af.CONFIRM_LABEL, sl)
            if sl in self.uids and not np.array_equal(self.uids[sl], u):
                raise ValueError("per-user vectors are not aligned")
            self.uids[sl] = u
            self.vectors[arm][seed][sl] = v
            if key in self.metrics:
                self.checks[f"{sl}/{key}"] = abs(
                    float(v.mean()) - self.metrics[key][af.CONFIRM_LABEL][sl]["ndcg@10"]
                )

    def run(self, arm: str, features: Sequence[str], seeds: Sequence[int]) -> dict:
        self.state["stage"] = f"fit {arm}"
        self.dog.reset_peak()
        t_arm = time.perf_counter()
        cfg0 = af.confirm_config(self.params, seeds[0])
        t0 = time.perf_counter()
        train, dropped = af.confirm_train(lambda n: self.datas[n].frame, features, cfg0)
        prep = time.perf_counter() - t0
        target = self.datas[self.sel].frame
        n_pos = ds.group_n_pos(self.datas, [self.sel], af.CONFIRM_LABEL)
        fits, gains = {}, None
        for seed in seeds:
            res, scores, info = af.fit_arm(train, target, n_pos, features, self.params, seed)
            key = f"{arm}:s{seed}"
            cfg = af.confirm_config(self.params, seed)
            t0 = time.perf_counter()
            self.metrics[key], self.all_users[key] = af.arm_metrics(
                self.datas[self.sel], key, scores, cfg
            )
            self.add_vectors(arm, seed, scores, key)
            info["metric_seconds"] = time.perf_counter() - t0
            info["dropped_zero_positive_groups"] = dropped
            if seed == seeds[0]:
                by_feature = gb.gain_importance(res)["by_feature"]
                gains = af.gain_breakdown(by_feature, self.catalog)
            fits[f"s{seed}"] = info
            print(
                f"  {key:16} best_iter {info['best_iteration']:4d} ndcg@10 "
                f"{info['best_ndcg@10']:.6f} fit {info['fit_seconds']:.0f} s "
                f"rss {process_rss() / MIB:.0f} MB",
                flush=True,
            )
            del res, scores
        del train
        self.state["completed"].append(arm)
        keys = [f"{arm}:s{s}" for s in seeds]
        return {
            "n_features": len(features),
            "features_sha256": af.list_sha256(features),
            "seeds": list(seeds),
            "fits": fits,
            "summary": summarize(self.metrics, keys),
            "gain_s42": gains,
            "seconds": {
                "prepare": prep,
                "fit_total": sum(f["fit_seconds"] for f in fits.values()),
                "predict_total": sum(f["predict_seconds"] for f in fits.values()),
                "arm_total": time.perf_counter() - t_arm,
            },
            "peak_rss_mb_sampled": self.dog.peak_bytes / MIB,
            "peak_rss_mb_lifetime": gb.peak_rss_mb(),
        }

    def mean_vector(self, arm: str, sl: str) -> np.ndarray:
        return np.mean([v[sl] for v in self.vectors[arm].values()], axis=0)


def twin_check(frame: pl.DataFrame, twins: Mapping) -> dict:
    out, bad = {}, []
    for kind, pairs in (("exact", twins["exact"]), ("close", twins["close"])):
        for g, h in sorted(pairs.items()):
            a = frame[g].cast(pl.Float32).fill_null(float("nan")).to_numpy()
            b = frame[h].cast(pl.Float32).fill_null(float("nan")).to_numpy()
            if kind == "exact":
                ok = af.bit_equal(a, b)
            else:
                ok = bool(np.allclose(a, b, rtol=1e-6, equal_nan=True))
            out[h] = {
                "generated": g,
                "kind": kind,
                "equal": ok,
                "rows": int(a.size),
                "missing": int(np.isnan(b).sum()),
            }
            if not ok:
                bad.append(f"{h}~{g}")
    if bad:
        raise AssertionError(f"hand twins differ from their generated twins on train_280: {bad}")
    return {
        "n_twins": len(out),
        "n_exact": len(twins["exact"]),
        "n_close": len(twins["close"]),
        "all_equal": True,
        "twins": out,
    }


def overlap_corr(
    datas, hand: Sequence[str], generated: Sequence[str], users: int, cache_dir: Path
) -> tuple[np.ndarray, dict]:
    sel = af.confirm_fold().valid
    npos = {sel: datas[sel].n_pos[af.CONFIRM_LABEL].select("uid", pl.col("all").alias("n_pos"))}
    base = datas[sel].frame
    zh, info = af.cluster_z(
        {sel: base.select(*ds.PAIR, af.CONFIRM_LABEL, *hand)}, npos, hand
    )
    corr = np.empty((len(hand), len(generated)), dtype=np.float64)
    for s in range(0, len(generated), OVERLAP_BATCH):
        names = list(generated[s : s + OVERLAP_BATCH])
        frame = af.confirm_frame(sel, names, users, cache_dir)
        if not frame.select(ds.PAIR).equals(base.select(ds.PAIR)):
            raise ValueError("train_280 rows changed between loads")
        zg, ginfo = af.cluster_z(
            {sel: frame.select(*ds.PAIR, af.CONFIRM_LABEL, *names)}, npos, names
        )
        if ginfo != info:
            raise ValueError("rho samples differ between batches")
        corr[:, s : s + len(names)] = zh.astype(np.float64).T @ zg.astype(np.float64)
        del frame, zg
    return corr, info


def read_r1(out_dir: Path, stem: str, code: int, wall: float) -> tuple[dict, dict | None]:
    done = out_dir / f"{stem}-autofeat-r1.json"
    partial = out_dir / f"{stem}-autofeat-r1-partial.json"
    vectors = out_dir / f"{stem}-autofeat-r1-per-user.parquet"
    if code == 0 and done.is_file():
        body = json.loads(done.read_text())
        table = pl.read_parquet(vectors)
        vecs = {sl: (table.filter(pl.col("slice") == sl)["uid"].to_numpy(),
                     table.filter(pl.col("slice") == sl)["ndcg@10"].to_numpy()) for sl in BOOT_SLICES}
        return {
            "status": "completed",
            "file": done.name,
            "per_user_file": vectors.name,
            "exit_code": code,
            "wall_seconds": wall,
            **{k: body[k] for k in ("arm", "metrics", "metrics_all_users", "resources", "runtime_seconds")},
        }, vecs
    if code == ABORT_EXIT_CODE and partial.is_file():
        return {
            "status": "aborted_rss_limit",
            "file": partial.name,
            "exit_code": code,
            "wall_seconds": wall,
            "partial": json.loads(partial.read_text()),
            "finding": "the full-grid fit exceeded the RSS limit; not retried with subsampling",
        }, None
    return {"status": "failed", "exit_code": code, "wall_seconds": wall}, None


def confirm_r1(args) -> int:
    stem = args.run_stem
    now = datetime.strptime(stem, STEM_FORMAT)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    state: dict = {"stage": "setup", "completed": [], "info": {"arm": R1_ARM}}
    dog = Watchdog(args.max_rss_gb * GIB, state, out_dir / f"{stem}-autofeat-r1-partial.json")
    dog.start()
    try:
        start = time.perf_counter()
        runtime: dict[str, float] = {}
        params, params_source = load_params(args.params_json)
        catalog = af.feature_catalog()
        fixed = list(af.fixed_features())
        features = [*fixed, *(s.name for s in af.generated_specs())]
        sel = af.confirm_fold().valid
        seed = af.R1_SEEDS[0]
        cfg = af.confirm_config(params, seed)
        loaded = []

        def load(name):
            state["stage"] = f"load {name}"
            loaded.append(name)
            return af.confirm_frame(name, features, args.users, cache_dir)

        t0 = time.perf_counter()
        train, dropped = af.confirm_train(load, features, cfg)
        runtime["load_train"] = time.perf_counter() - t0
        state["info"]["train_rows"] = train.height
        print(f"R1 train {train.shape} rss {process_rss() / MIB:.0f} MB", flush=True)
        state["stage"] = f"load {sel}"
        t0 = time.perf_counter()
        data = af.confirm_window_data(sel, features, args.users, cache_dir)
        loaded.append(sel)
        target = data.frame.select(*ds.KEYS, *features, af.CONFIRM_LABEL)
        n_pos = ds.group_n_pos({sel: data}, [sel], af.CONFIRM_LABEL)
        runtime["load_target"] = time.perf_counter() - t0
        print(f"R1 {sel} {target.shape} rss {process_rss() / MIB:.0f} MB", flush=True)
        state["stage"] = "fit"
        res, scores, info = af.fit_arm(train, target, n_pos, features, params, seed)
        del train, target
        state["completed"].append("fit")
        state["stage"] = "metrics"
        key = f"{R1_ARM}:s{seed}"
        metrics, all_users = af.arm_metrics(data, key, scores, cfg)
        gains = af.gain_breakdown(gb.gain_importance(res)["by_feature"], catalog)
        rows = []
        for sl in BOOT_SLICES:
            u, v = af.per_user_ndcg(data.frame, scores, data.n_pos[af.CONFIRM_LABEL], af.CONFIRM_LABEL, sl)
            rows.append(pl.DataFrame({"slice": [sl] * u.size, "uid": u, "ndcg@10": v}))
        ds._write_atomic(pl.concat(rows), out_dir / f"{stem}-autofeat-r1-per-user.parquet")
        runtime["total"] = time.perf_counter() - start
        record = {
            "seed": seed,
            "seeds": list(af.R1_SEEDS),
            "git_commit": gb.GIT_COMMIT,
            "stage": "confirm-r1",
            "protocol": "inner",
            "selection_window": sel,
            "loaded_windows": loaded,
            "config": {
                "version": af.CONFIRM_VERSION,
                "params": params,
                "params_source": params_source,
                "fit_config": cfg.config_dict(),
                "rule": CONFIRM_RULES["r1"],
            },
            "arm": {
                "name": R1_ARM,
                "n_features": len(features),
                "n_generated": len(features) - len(fixed),
                "features_sha256": af.list_sha256(features),
                "fit": {**info, "dropped_zero_positive_groups": dropped},
                "gain_s42": gains,
            },
            "metrics": {sel: {lb: {sl: {key: metrics[lb][sl]} for sl in SLICES} for lb in LABELS}},
            "metrics_all_users": {
                sel: {lb: {sl: {key: all_users[lb][sl]} for sl in SLICES} for lb in LABELS}
            },
            "runtime_seconds": runtime,
            "resources": {
                "peak_rss_mb_lifetime": gb.peak_rss_mb(),
                "peak_rss_mb_sampled": dog.peak_bytes / MIB,
                "max_rss_gb": args.max_rss_gb,
            },
        }
        af.assert_no_key(record)
        path, _ = gb.write_run_log(record, "autofeat-r1", out_dir, now)
        print(
            f"R1 ndcg@10 {metrics['y_org']['all']['ndcg@10']:.6f} best_iter "
            f"{info['best_iteration']} fit {info['fit_seconds']:.0f} s peak_rss "
            f"{gb.peak_rss_mb():.0f} MB, log {path}",
            flush=True,
        )
        return 0
    finally:
        dog.stop()


def e1_reference(runs_dir: Path) -> tuple[Path, dict]:
    path = newest_log(runs_dir, "*-exp-E1.json")
    return path, json.loads(path.read_text())


def prereg_reference(e1_path: Path, e1: Mapping, hand: Sequence[str]) -> dict:
    per_seed = {}
    for s in af.PREREG_SEEDS:
        key = f"E1a:s{s}"
        fit = e1["models"][key]["fit"]
        per_seed[f"s{s}"] = {
            "inner_train_280_ndcg@10": e1["metrics"]["inner"]["train_280"]["y_org"]["all"][key]["ndcg@10"],
            "val_ndcg@10": e1["metrics"]["outer"]["val"]["y_org"]["all"][key]["ndcg@10"],
            "best_iteration": fit["inner"]["best_iteration"],
            "num_boost_round": fit["outer"]["num_boost_round"],
        }
    return {
        "arm": "E1a",
        "features": list(hand),
        "sha256": af.list_sha256(hand),
        "source_log": project_path(e1_path),
        "source_log_sha256": af.file_sha256(e1_path),
        "rounds_rule": "Phase 2 outer protocol: each seed refits with round(best_iteration x 1.5) "
        "of its own inner fit, as logged; the val stage must reproduce the logged per-seed "
        "values to 6 decimals",
        "per_seed": per_seed,
    }


def stage_confirm(args) -> int:
    if args.r1_only:
        if not args.run_stem:
            raise ValueError("--r1-only needs --run-stem")
        return confirm_r1(args)
    now = datetime.now()
    stem = f"{now:{STEM_FORMAT}}"
    out_dir = Path(args.out_dir)
    for name in ("autofeat-inner", "autofeat-prereg", "autofeat-r1"):
        if (out_dir / f"{stem}-{name}.json").exists():
            raise FileExistsError(f"{stem}-{name}.json already exists")
    state: dict = {"stage": "setup", "completed": [], "info": {"stem": stem}}
    dog = Watchdog(args.max_rss_gb * GIB, state, out_dir / f"{stem}-autofeat-confirm-partial.json")
    dog.start()
    try:
        return run_confirm(args, now, stem, state, dog)
    finally:
        dog.stop()


def run_confirm(args, now: datetime, stem: str, state: dict, dog: Watchdog) -> int:
    start = time.perf_counter()
    runtime: dict[str, float] = {}
    out_dir, runs_dir, cache_dir = Path(args.out_dir), Path(args.runs_dir), Path(args.cache_dir)
    fold = af.confirm_fold()
    sel = fold.valid
    seeds = list(af.CONFIRM_SEEDS)
    params, params_source = load_params(args.params_json)
    catalog = af.feature_catalog()
    specs = af.hand_specs()
    hand = list(gb.resolve_features(gb.GBDTConfig(params=params), specs))
    group_of = {s.name: s.group for s in specs}
    fixed = list(af.fixed_features())
    lists, list_meta = read_lists(runs_dir)
    select_path = newest_log(
        runs_dir,
        "*-autofeat-select.json",
        lambda b: all(b["sha256"][k]["sorted"] == v["sha256"] for k, v in list_meta.items()),
    )
    select = json.loads(select_path.read_text())
    screen = json.loads((runs_dir / select["config"]["screen_log"]["file"]).read_text())
    t0 = time.perf_counter()
    r3b_lists, r3b_info = r3b_from_log(select, screen, lists)
    runtime["r3b_from_log"] = time.perf_counter() - t0
    r3b_files = {}
    for k in af.TOP_KS:
        key = f"R3b_K{k}"
        rec = af.post_hoc_record(af.POST_HOC_METHOD, k, r3b_lists[key], stem, CONFIRM_RULES["r3b"])
        path = out_dir / f"autofeat-select-R3b-K{k}.json"
        r3b_files[key] = {"file": path.name, "status": write_list_file(path, rec), "sha256": rec["sha256"]}
        lists[key] = r3b_lists[key]
    print(f"R3b (POST-HOC) lists from {select_path.name}: " + json.dumps(r3b_files), flush=True)
    if lists["R4_K114"] != lists["R4_K40"]:
        raise RuntimeError("R4 K114 and K40 lists differ; the tiny branch expects one list")
    arm_gen = {
        R0_ARM: [],
        **{f"{m}_K{k}": lists[f"{m}_K{k}"] for m in ("R2", "R3") for k in af.TOP_KS},
        R4_TINY: lists["R4_K114"],
        **{f"R3b_K{k}": lists[f"R3b_K{k}"] for k in af.TOP_KS},
    }
    arm_features = {a: (hand if a == R0_ARM else [*fixed, *g]) for a, g in arm_gen.items()}
    union = list(dict.fromkeys(c for f in arm_features.values() for c in f))
    reference = r0_reference(runs_dir)
    e1_path, e1 = e1_reference(runs_dir)
    state["stage"] = "load"
    t0 = time.perf_counter()
    uids = ds.sample_users(args.data, args.users)
    events, data_info = af.load_selection_events(args.data, uids)
    if data_info["loaded_max_ts"] >= VAL_TS:
        raise RuntimeError("loaded events reach day 287")
    runtime["load_events"] = time.perf_counter() - t0
    print(
        f"loaded {data_info['loaded_rows']} events, max timestamp {data_info['loaded_max_ts']} "
        f"< {VAL_TS} (day 287)",
        flush=True,
    )
    loaded: list[str] = []
    datas = {}
    t0 = time.perf_counter()
    for w in af.confirm_window_names():
        datas[w] = af.confirm_window_data(w, union, args.users, cache_dir)
        loaded.append(w)
        print(f"  loaded {w} {datas[w].frame.shape} rss {process_rss() / MIB:.0f} MB", flush=True)
    runtime["load_windows"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bframe, _ = gb.add_baselines(events, datas[sel], gb.eval_frame(datas[sel]))
    del events
    gc.collect()
    runtime["baselines"] = time.perf_counter() - t0
    state["stage"] = "overlap"
    t0 = time.perf_counter()
    twins = hand_twins()
    twin_cols = list(dict.fromkeys([*twins["exact"], *twins["exact"].values(), *twins["close"], *twins["close"].values()]))
    twin_res = twin_check(af.confirm_frame(sel, twin_cols, args.users, cache_dir), twins)
    print(f"twins bit-equal on {sel}: {twin_res['n_exact']} exact + {twin_res['n_close']} close", flush=True)
    generated = [s.name for s in af.generated_specs()]
    corr_hg, overlap_sample = overlap_corr(datas, hand, generated, args.users, cache_dir)
    runtime["overlap_data"] = time.perf_counter() - t0
    print(f"spearman hand x grid on {overlap_sample} rss {process_rss() / MIB:.0f} MB", flush=True)
    runner = ArmRunner(datas, params, catalog, dog, state)
    arms: dict[str, dict] = {}
    for arm in FIRST_ARMS:
        print(f"arm {arm}: {len(arm_features[arm])} features", flush=True)
        arms[arm] = runner.run(arm, arm_features[arm], seeds)
        if arm == R0_ARM:
            value = runner.metrics[f"{R0_ARM}:s{gb.SEED}"]["y_org"]["all"]["ndcg@10"]
            if args.users == 0 and (reference is None or abs(value - reference["value"]) > R0_TOL):
                print(f"R0 MISMATCH: {value:.6f} vs {reference}", file=sys.stderr)
                return R0_EXIT_CODE
    r0_logged = e1["metrics"]["inner"]["train_280"]
    r0_check = {
        "reference": None
        if reference is None
        else {"source": Path(reference["log"]).name, "key": reference["key"], "ndcg@10": reference["value"]},
        "tolerance": R0_TOL,
        "per_seed": {},
    }
    for s in seeds:
        mine = runner.metrics[f"{R0_ARM}:s{s}"]
        theirs = {lb: {sl: r0_logged[lb][sl][f"E1a:s{s}"] for sl in SLICES} for lb in LABELS}
        diffs = [
            abs(mine[lb][sl][m] - theirs[lb][sl][m])
            for lb in LABELS
            for sl in SLICES
            for m in theirs[lb][sl]
            if theirs[lb][sl][m] is not None and mine[lb][sl].get(m) is not None
        ]
        r0_check["per_seed"][f"s{s}"] = {
            "ndcg@10": mine["y_org"]["all"]["ndcg@10"],
            "logged_ndcg@10": theirs["y_org"]["all"]["ndcg@10"],
            "delta": mine["y_org"]["all"]["ndcg@10"] - theirs["y_org"]["all"]["ndcg@10"],
            "max_abs_diff_all_logged_metrics": max(diffs),
        }
    r0_check["match_s42"] = abs(r0_check["per_seed"]["s42"]["delta"]) <= R0_TOL
    r0_check["source_log"] = e1_path.name
    choice_scores = {
        m: [runner.metrics[f"{a}:s{s}"]["y_org"]["all"]["ndcg@10"] for s in seeds]
        for m, a in (("R2", "R2_K114"), ("R3", "R3_K114"), ("R4", R4_TINY), ("R3b", "R3b_K114"))
    }
    choice = af.choose_method(choice_scores)
    winner = choice["chosen"]
    g127, g53 = f"{winner}_K114", f"{winner}_K40"
    print(f"method choice: {winner} ({choice['reason']}); means {choice['means']} stds {choice['stds']}", flush=True)
    state["stage"] = "near copies"
    t0 = time.perf_counter()
    win = lists[g127]
    d259 = af.select_window_data("train_259", [*hand, *win], args.users, cache_dir)
    loaded.insert(0, "train_259")
    rho_frames = {
        "train_259": d259.frame.select(*ds.PAIR, af.CONFIRM_LABEL, *hand, *win),
        "train_266": datas["train_266"].frame.select(*ds.PAIR, af.CONFIRM_LABEL, *hand, *win),
    }
    rho_npos = {
        "train_259": d259.n_pos[af.CONFIRM_LABEL].select("uid", pl.col("all").alias("n_pos")),
        "train_266": datas["train_266"].n_pos[af.CONFIRM_LABEL].select("uid", pl.col("all").alias("n_pos")),
    }
    z, near_sample = af.cluster_z(rho_frames, rho_npos, [*win, *hand])
    del d259, rho_frames
    corr_wh = af.rho_matrix(z)[: len(win), len(win) :]
    del z
    copies = af.near_copies(corr_wh, win, hand)
    if args.users == 0 and {k: near_sample[k] for k in ("users", "rows")} != {
        k: select["clusters"]["sample"][k] for k in ("users", "rows")
    }:
        raise RuntimeError("near-copy sample differs from the screening rho sample")
    u_gen = [g for g in win if g not in copies]
    r5_features = af.union_features(hand, win, copies)
    runtime["near_copies"] = time.perf_counter() - t0
    print(f"R5: {len(copies)} near-copies removed, {len(r5_features)} features", flush=True)
    arms[R5_ARM] = runner.run(R5_ARM, r5_features, seeds)
    arm_gen[R5_ARM] = u_gen
    arm_features[R5_ARM] = r5_features
    b_vectors = {}
    for b in REFERENCE_BASELINES:
        b_vectors[b] = {}
        for sl in BOOT_SLICES:
            u, v = af.per_user_ndcg(datas[sel].frame, bframe[b].to_numpy(), datas[sel].n_pos[af.CONFIRM_LABEL], af.CONFIRM_LABEL, sl)
            if not np.array_equal(u, runner.uids[sl]):
                raise ValueError("baseline per-user vectors are not aligned")
            b_vectors[b][sl] = v
    baseline_check = {
        b: {
            sl: abs(float(b_vectors[b][sl].mean()) - r0_logged["y_org"][sl][b]["ndcg@10"])
            for sl in BOOT_SLICES
        }
        for b in REFERENCE_BASELINES
    }
    state["stage"] = "overlap analysis"
    best_full = af.max_abs_rho(corr_hg, hand, generated)
    idx = {n: j for j, n in enumerate(generated)}
    best_win = af.max_abs_rho(corr_hg[:, [idx[n] for n in win]], hand, win)
    hand_to_twin = {h: g for g, h in twins["exact"].items()}
    h3 = af.h3_check(HAND_TOP10, hand_to_twin, best_full, win, catalog)
    h4 = af.h4_check(
        [runner.metrics[f"{g127}:s{s}"]["y_org"]["all"]["ndcg@10"] for s in seeds],
        [runner.metrics[f"{g53}:s{s}"]["y_org"]["all"]["ndcg@10"] for s in seeds],
    )
    h5 = af.h5_check(arms[g127]["gain_s42"])
    overlap = {
        "sample": overlap_sample,
        "twins": twin_res,
        "max_abs_rho_full_grid": best_full,
        "max_abs_rho_winning_k114": best_win,
        "group_recovery_full_grid": af.group_recovery(best_full, group_of),
        "group_recovery_winning_k114": af.group_recovery(best_win, group_of),
        "gain_g127_s42": arms[g127]["gain_s42"],
        "hypotheses": {
            "H3": h3,
            "H4": h4,
            "H5": h5,
            "not_computed": "H1 and H2 belong to the pre-registered next stage",
        },
    }
    sel_frame_rows = datas[sel].frame.height
    n_hist = datas[sel].n_hist_users
    del datas, runner.datas, bframe
    gc.collect()
    r1, r1_vectors = {"status": "skipped", "reason": "--skip-r1 (fast rerun)"}, None
    if not args.skip_r1:
        state["stage"] = "R1 subprocess"
        print(f"R1 in a separate process, rss of this process {process_rss() / MIB:.0f} MB", flush=True)
        cmd = [
            sys.executable, "-I", str(Path(__file__).resolve()), "--stage", "confirm",
            "--r1-only", "--run-stem", stem, "--users", str(args.users),
            "--cache-dir", str(cache_dir), "--out-dir", str(out_dir),
            "--runs-dir", str(runs_dir), "--params-json", str(args.params_json),
            "--max-rss-gb", str(args.max_rss_gb), "--data", str(args.data),
        ]
        t0 = time.perf_counter()
        code = subprocess.run(cmd, check=False).returncode
        r1, r1_vectors = read_r1(out_dir, stem, code, time.perf_counter() - t0)
        runtime["r1_process"] = time.perf_counter() - t0
        print(f"R1 status {r1['status']} exit {code}", flush=True)
    state["stage"] = "bootstrap"
    means = {a: {sl: runner.mean_vector(a, sl) for sl in BOOT_SLICES} for a in arms}
    if r1_vectors is not None:
        for sl in BOOT_SLICES:
            if not np.array_equal(r1_vectors[sl][0], runner.uids[sl]):
                raise ValueError("R1 per-user vectors are not aligned")
        means[R1_ARM] = {sl: r1_vectors[sl][1] for sl in BOOT_SLICES}
    boot = {}
    for sl in BOOT_SLICES:
        boot[sl] = {}
        for a in means:
            if a != R0_ARM:
                boot[sl][f"{a} vs {R0_ARM}"] = af.paired_bootstrap(means[a][sl], means[R0_ARM][sl])
            boot[sl][f"{a} vs B3"] = af.paired_bootstrap(means[a][sl], b_vectors["B3"][sl])
    labels = {
        R0_ARM: ("reference", "hand 127 (E1a)"),
        "R2_K114": ("eligible", "R2 screening top 114 + C/Z"),
        "R2_K40": ("budget", "R2 screening top 40 + C/Z"),
        "R3_K114": ("eligible", "R3 stability, cap 3 per cluster, 114 + C/Z"),
        "R3_K40": ("budget", "R3 stability, cap 3 per cluster, 40 + C/Z"),
        R4_TINY: ("tiny branch, not eligible", "R4 greedy, 4 features, no padding, + C/Z"),
        "R3b_K114": ("POST-HOC, not eligible", "R3 without the cluster cap, 114 + C/Z"),
        "R3b_K40": ("POST-HOC, not eligible", "R3 without the cluster cap, 40 + C/Z"),
        R5_ARM: ("secondary", f"hand 127 + {g127} minus near-copies"),
    }
    list_files = {a: list_meta[a]["file"] for a in arms if a in list_meta}
    list_files[R4_TINY] = list_meta["R4_K114"]["file"]
    list_files.update({a: r3b_files[a]["file"] for a in r3b_files})
    for a, rec in arms.items():
        rec["role"], rec["description"] = labels[a]
        rec["post_hoc"] = a.startswith("R3b")
        rec["eligible_for_choice"] = a in ("R2_K114", "R3_K114")
        rec["generated"] = list(arm_gen[a])
        rec["list_file"] = list_files.get(a)
    keys = list(runner.metrics)
    metrics = {sel: {lb: {sl: {k: runner.metrics[k][lb][sl] for k in keys} for sl in SLICES} for lb in LABELS}}
    all_users = {sel: {lb: {sl: {k: runner.all_users[k][lb][sl] for k in keys} for sl in SLICES} for lb in LABELS}}
    for b in REFERENCE_BASELINES:
        for lb in LABELS:
            for sl in SLICES:
                metrics[sel][lb][sl][b] = r0_logged[lb][sl][b]
                all_users[sel][lb][sl][b] = e1["metrics_all_users"]["inner"]["train_280"][lb][sl][b]
    stats = {a: gb.seed_stats(metrics, [f"{a}:s{s}" for s in seeds])[sel] for a in arms}
    n_fits = sum(len(r["seeds"]) for r in arms.values()) + (1 if r1["status"] != "skipped" else 0)
    bookkeeping = {
        "arms_on_s3": [*arms, *([R1_ARM] if r1["status"] != "skipped" else [])],
        "n_arms": len(arms) + (1 if r1["status"] != "skipped" else 0),
        "n_fits": n_fits,
        "eligible_for_choice": ["R2_K114", "R3_K114"],
        "not_eligible": [R4_TINY, "R3b_K114", "R3b_K40", R5_ARM, R1_ARM, "R2_K40", "R3_K40"],
        "post_hoc": ["R3b_K114", "R3b_K40"],
        "earlier_uses_of_train_280": "Phase 2 tuning grid (8 points), E1-E5 inner choices, "
        "autofeat build R0 reproduction",
    }
    prereg_status, prereg_path, binding = "skipped", None, None
    if args.users == 0:
        arm_rec = {}
        for name, a, role in (("G127", g127, "primary"), ("G53", g53, "secondary"), ("U", R5_ARM, "secondary")):
            fits = arms[a]["fits"]
            arm_rec[name] = af.prereg_arm(
                role,
                arm_features[a],
                arm_gen[a],
                fits["s42"]["best_iteration"],
                list_file=list_meta[a]["file"] if a in list_meta else None,
                derived_from=None if a in list_meta else f"hand 127 (registry order) + {g127} "
                f"minus {len(copies)} near-copies (|Spearman| > {af.NEAR_COPY_RHO} on the "
                "train_259 + train_266 screening sample)",
                info_best_iterations={k: f["best_iteration"] for k, f in fits.items() if k != "s42"},
            )
        binding = af.prereg_binding(
            winner,
            arm_rec,
            prereg_reference(e1_path, e1, hand),
            project_path(args.params_json),
            params_source["sha256"],
        )
        digest = af.binding_digest(binding)
        existing = sorted(out_dir.glob("*-autofeat-prereg.json"))
        if existing:
            prereg_path = existing[-1]
            old = json.loads(prereg_path.read_text())
            prereg_status = "identical" if old.get("decision_digest") == digest else "conflict"
        else:
            prereg_path = out_dir / f"{stem}-autofeat-prereg.json"
            prereg_status = "new"
    allowed = {*af.NEAR_COPY_WINDOWS, *af.confirm_window_names()}
    if not set(loaded) <= allowed or set(loaded) & set(af.FORBIDDEN):
        raise RuntimeError(f"confirm read windows {loaded}, allowed {sorted(allowed)}")
    af.assert_selection_horizon(loaded, af.CONFIRM_LIMIT)
    runtime["total_before_logs"] = time.perf_counter() - start
    record = {
        "seed": gb.SEED,
        "seeds": seeds,
        "git_commit": gb.GIT_COMMIT,
        "stage": "confirm",
        "protocol": "inner",
        "selection_window": sel,
        "loaded_windows": [w for w in af.WINDOW_NAMES if w in set(loaded)],
        "config": {
            "version": af.CONFIRM_VERSION,
            "fold": {
                "name": fold.name,
                "train": list(fold.train),
                "holdout_window": fold.valid,
                "role": fold.role,
            },
            "confirm_limit_day": af.CONFIRM_LIMIT / DAY,
            "params": params,
            "params_source": params_source,
            "fit_config": af.confirm_config(params, gb.SEED).config_dict(),
            "seeds": seeds,
            "r1_seeds": list(af.R1_SEEDS),
            "users_sample": {"n": args.users, "seed": gb.SEED},
            "lists": list_meta,
            "select_log": select_path.name,
            "screen_log": select["config"]["screen_log"]["file"],
            "e1_log": e1_path.name,
            "rules": CONFIRM_RULES,
            "max_rss_gb": args.max_rss_gb,
        },
        "data": {**data_info, "rows_train_280": sel_frame_rows, "n_hist_users_train_280": n_hist},
        "versions": gb.library_versions(),
        "r0_check": r0_check,
        "baseline_check": {
            "rule": "abs(mean per-user ndcg@10 from the baseline code - logged inner train_280 value)",
            "diffs": baseline_check,
            "source": e1_path.name,
        },
        "r3b": {"post_hoc": True, "files": r3b_files, **r3b_info},
        "arms": arms,
        "method_choice": {**choice, "G127": g127, "G53": g53, "U": R5_ARM},
        "near_copies": {"sample": near_sample, "threshold": af.NEAR_COPY_RHO, "removed": copies, "kept_generated": len(u_gen)},
        "r1": r1,
        "metrics_layout": "metrics[split][label][slice][arm:seed] -> metric dict; B3 and B5 copied from the E1 log",
        "metrics": metrics,
        "metrics_all_users": all_users,
        "seed_stats": stats,
        "per_user_check": runner.checks,
        "bootstrap": {"rule": CONFIRM_RULES["bootstrap"], "label": af.CONFIRM_LABEL, "metric": "ndcg@10", "slices": boot},
        "overlap": overlap,
        "s3_bookkeeping": bookkeeping,
        "prereg": {
            "file": None if prereg_path is None else prereg_path.name,
            "status": prereg_status,
            "decision_digest": None if binding is None else af.binding_digest(binding),
        },
    }
    runtime["total"] = time.perf_counter() - start
    record["runtime_seconds"] = runtime
    record["resources"] = {
        "peak_rss_mb_lifetime": gb.peak_rss_mb(),
        "peak_rss_mb_per_arm": {a: r["peak_rss_mb_sampled"] for a, r in arms.items()},
    }
    af.assert_no_key(record)
    inner_path, inner_digest = gb.write_run_log(record, "autofeat-inner", out_dir, now)
    af.assert_no_key(json.loads(inner_path.read_text()))
    verify = None
    if prereg_status == "new":
        prereg = af.prereg_record(
            binding,
            {
                "file": inner_path.name,
                "sha256": af.file_sha256(inner_path),
                "metrics_digest": inner_digest,
                "r1_file": r1.get("file"),
            },
        )
        written, _ = gb.write_run_log(prereg, "autofeat-prereg", out_dir, now)
        if written != prereg_path:
            raise RuntimeError(f"prereg written to {written}, expected {prereg_path}")
    if prereg_path is not None:
        verify = af.verify_prereg(prereg_path)
    print_confirm(record, choice, boot, overlap, arms, r1)
    print(f"inner log: {inner_path}")
    print(f"metrics_digest: {inner_digest}")
    if prereg_path is not None:
        print(f"prereg: {prereg_path} status {prereg_status} digest {record['prereg']['decision_digest']}")
        for name, arm in binding["arms"].items():
            print(f"  {name}: {arm['n_features']} features sha256 {arm['sha256']} rounds {arm['num_boost_round']}")
        print(f"verify_prereg: ok {verify['ok']} diff {verify['diff']}")
    print("timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in runtime.items()))
    if prereg_status == "conflict" or (verify is not None and not verify["ok"]):
        print("PREREG CONFLICT or verification failure", file=sys.stderr)
        return 1
    if any(v["status"] == "conflict" for v in r3b_files.values()):
        print("LIST CONFLICT: an existing R3b list file differs", file=sys.stderr)
        return 1
    return 0


def print_confirm(record, choice, boot, overlap, arms, r1) -> None:
    sel = record["selection_window"]
    print(f"S3 on {sel} (y_org): arm, n_feat, ndcg@10 mean+-std, recall@10, recall_cap@10, novel ndcg@10")
    for a, r in arms.items():
        s = r["summary"]
        print(
            f"  {a:10} {r['n_features']:4d} {s['y_org/all/ndcg@10']['mean']:.4f}+-"
            f"{s['y_org/all/ndcg@10']['std'] or 0.0:.4f} {s['y_org/all/recall@10']['mean']:.4f} "
            f"{s['y_org/all/recall_cap@10']['mean']:.4f} {s['y_org/novel/ndcg@10']['mean']:.4f}"
            f"  {r['role']}"
        )
    if r1["status"] == "completed":
        m = r1["metrics"][sel]["y_org"]
        k = next(iter(m["all"]))
        print(
            f"  {R1_ARM:10} {r1['arm']['n_features']:4d} {m['all'][k]['ndcg@10']:.4f} (seed 42) "
            f"{m['all'][k]['recall@10']:.4f} {m['all'][k]['recall_cap@10']:.4f} {m['novel'][k]['ndcg@10']:.4f}"
        )
    for b in REFERENCE_BASELINES:
        m = record["metrics"][sel]["y_org"]
        print(
            f"  {b:10}      {m['all'][b]['ndcg@10']:.4f} {m['all'][b]['recall@10']:.4f} "
            f"{m['all'][b]['recall_cap@10']:.4f} {m['novel'][b]['ndcg@10']:.4f}"
        )
    print("bootstrap ndcg@10 (y_org, seed-averaged per user):")
    for sl, pairs in boot.items():
        for name, r in pairs.items():
            print(f"  {sl:6} {name:22} {r['mean_delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    hyp = overlap["hypotheses"]
    print(f"H3 holds {hyp['H3']['holds']} covered {hyp['H3']['n_covered']}/10 decay {hyp['H3']['pair_decay_in_g127']}")
    print(f"H4 holds {hyp['H4']['holds']} difference {hyp['H4']['difference']:.4f}")
    print(f"H5 holds {hyp['H5']['holds']} pair share {hyp['H5']['pair_share_of_total_gain']:.3f}")
    rec = overlap["group_recovery_full_grid"]
    print("hand groups recovered (full grid, |rho| > 0.95): " + ", ".join(
        f"{g} {v['n_recovered']}/{v['n_features']}" for g, v in rec.items()))


VAL_VERSION = "autofeat-val-1"
VAL_WINDOW = "val"
VAL_GUARD_NAME = "autofeat-val.guard"
VAL_GUARD_EXIT_CODE = 4
VAL_PREREG_EXIT_CODE = 5
VAL_REFERENCE = "E1a"
VAL_DIGITS = bl.DIGITS
VAL_PER_USER_TOL = 1e-9
VAL_ARM_KEYS = (
    "role",
    "features",
    "n_features",
    "num_boost_round",
    "best_iteration_s42",
    "best_iteration_other_seeds",
)
VAL_OTHER_SEEDS = ("s43", "s44")
INNER_LOG_ARMS = {"G127": "R3_K114", "G53": "R3_K40", "U": "R5_union"}
VAL_REFERENCE_KEYS = ("val_ndcg@10", "inner_train_280_ndcg@10", "best_iteration", "num_boost_round")
VAL_RULES = {
    "gate": "the stage refuses to start when a runs/*-autofeat-val.json or the guard marker "
    f"runs/{VAL_GUARD_NAME} exists, then verifies the pre-registration (autofeat.verify_prereg "
    "plus the binding checks, out-dir = runs-dir = prereg folder), loads the train windows "
    "from the autofeat manifest, builds the val candidates and feature parts from events "
    "before day 287 and runs every label-free inner fit (E1a seeds 42-46 and the three arms "
    "x 5 seeds); any difference aborts with exit 5 for E1a (best iteration, round count and "
    f"inner ndcg@10 to {VAL_DIGITS} decimals, seeds 42-46) and for the arms (best iteration "
    f"and inner ndcg@10 to {VAL_DIGITS} decimals against the inner log pinned in the "
    "pre-registration evidence, seeds 42-44 only); arm seeds 45 and 46 have no "
    "pre-registered reference, are logged and are not gated; the guard marker is created "
    "exclusively only after this gate and before any event at or after day 287 is read, so a "
    "crash cannot allow a silent second run; after the guard only the val labels, the refits "
    "and the val predictions run; the feature part file names used after the guard are "
    "compared with the label-free preparation, a difference only warns and is logged as "
    "post_guard_parts_match",
    "data": "train_266, train_273 and train_280 from the autofeat manifest; val candidates, "
    "labels and features are built in this stage from events with timestamp < day 294 "
    "(protocol outer, no test event is read); hand part = the 127 registered specs, "
    "generated parts = the full GU/GI/GD/GA spec lists, the same per-part builders and cache "
    "keys as the train windows",
    "reference": "E1a = gbdt.run_protocol(outer) with the 127 hand features, the frozen params "
    "file and the Phase 2 outer rule round(best_iteration x 1.5) of the seed's own inner fit; "
    "all 5 val seed values of ndcg@10 (y_org, slice all) must equal the logged values after "
    f"rounding to {VAL_DIGITS} decimals, otherwise the stage writes the discrepancy and no verdict",
    "arms": "G127, G53, U exactly as pre-registered (feature lists in the pre-registered order); "
    "per seed an inner fit (fit train_266 + train_273, early stopping on train_280) supplies "
    "the calibration scores for LogLoss and is logged against the pre-registered best "
    "iterations and inner ndcg@10 (seeds 42-44; seeds 45 and 46 are logged only); the val scores come from the refit on train_266 + train_273 + train_280 "
    "without early stopping with the pre-registered num_boost_round of the arm for every seed",
    "metrics": "all Phase 2 metrics on val (y_org and y_all, slices all and novel, all-users "
    "variants); probabilities for LogLoss from the percentile calibrator fitted on train_280 "
    "inner scores, as the Phase 2 outer protocol",
    "bootstrap": "paired user bootstrap on val, per-user ndcg@10 (y_org, slice all) averaged "
    "over the 5 seeds, users with >= 1 positive of the slice, 1000 resamples, numpy "
    "default_rng(42) restarted for every comparison; G127 vs E1a at the 95% level (H1), "
    "G53 vs E1a and U vs E1a at the 98.3% level; the novel slice is descriptive only",
    "decision": "H1 holds when the lower bound of the 95% CI of ndcg@10(G127) - ndcg@10(E1a) is "
    "> -0.005; H2 holds when the lower bound of the 98.3% CI of ndcg@10(U) - ndcg@10(E1a) is "
    "> 0; G53 is reported with its 98.3% CI; H3, H4 and H5 are not val-time items of the "
    "pre-registration",
}


class ValGuardError(RuntimeError):
    pass


def val_guard_hits(out_dir: str | Path, runs_dir: str | Path) -> list[str]:
    found: list[str] = []
    for d in dict.fromkeys(Path(x) for x in (out_dir, runs_dir)):
        marker = d / VAL_GUARD_NAME
        if marker.exists():
            found.append(str(marker))
        found.extend(str(p) for p in sorted(d.glob(af.VAL_LOG_GLOB)))
    return found


def claim_val_guard(out_dir: str | Path, info: Mapping) -> Path:
    path = Path(out_dir) / VAL_GUARD_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise ValGuardError(f"{path} already exists") from exc
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(bl._clean(info), indent=2) + "\n")
    return path


def resolve_prereg(runs_dir: str | Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        return path
    found = sorted(Path(runs_dir).glob("*-autofeat-prereg.json"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected exactly one *-autofeat-prereg.json in {runs_dir}, found {len(found)}; "
            "pass --prereg"
        )
    return found[0]


def check_binding(
    binding: Mapping,
    hand: Sequence[str],
    known: Sequence[str],
    params_sha256: str,
    root: Path | None = None,
) -> list[str]:
    root = af.PROJECT_ROOT if root is None else Path(root)
    problems: list[str] = []
    arms = binding.get("arms", {})
    if set(arms) != set(af.PREREG_ARMS):
        problems.append(f"arms {sorted(arms)} differ from {list(af.PREREG_ARMS)}")
    if list(binding.get("val_seeds", [])) != list(af.PREREG_SEEDS):
        problems.append("val_seeds differ from the code constants")
    if binding.get("label") != af.CONFIRM_LABEL or binding.get("objective") != "lambdarank":
        problems.append("label or objective differ from the code constants")
    if binding.get("round_factor") != gb.REFIT_FACTOR:
        problems.append("round_factor differs from gbdt.REFIT_FACTOR")
    if binding.get("params", {}).get("sha256") != params_sha256:
        problems.append("the params file in use differs from the pre-registered sha256")
    primary = binding.get("primary", {})
    if (
        primary.get("metric"),
        primary.get("label"),
        primary.get("slice"),
        primary.get("split"),
    ) != (af.PREREG_METRIC, af.CONFIRM_LABEL, "all", VAL_WINDOW):
        problems.append("primary metric, label, slice or split differ from the code constants")
    boot = binding.get("bootstrap", {})
    if (boot.get("n_boot"), boot.get("seed")) != (af.N_BOOT, af.BOOT_SEED):
        problems.append("bootstrap n_boot or seed differ from the code constants")
    hyp = binding.get("hypotheses", {})
    expected = (
        (hyp.get("H1", {}).get("ci_level"), af.PREREG_PRIMARY_CI, "H1 ci_level"),
        (hyp.get("H1", {}).get("margin"), af.PREREG_MARGIN, "H1 margin"),
        (hyp.get("H2", {}).get("ci_level"), af.PREREG_SECONDARY_CI, "H2 ci_level"),
        (hyp.get("secondary", {}).get("ci_level"), af.PREREG_SECONDARY_CI, "secondary ci_level"),
        (hyp.get("H1", {}).get("arm"), "G127", "H1 arm"),
        (hyp.get("H2", {}).get("arm"), "U", "H2 arm"),
    )
    for got, want, label in expected:
        if got != want:
            problems.append(f"{label} {got!r} differs from {want!r}")
    if list(hyp.get("secondary", {}).get("arms", [])) != ["G53", "U"]:
        problems.append("secondary arms differ from G53, U")
    known_set = set(known)
    for name, arm in arms.items():
        feats = arm.get("features", [])
        if len(set(feats)) != len(feats) or len(feats) != arm.get("n_features"):
            problems.append(f"arm {name}: feature list is not unique or n_features differs")
        missing = sorted(set(feats) - known_set)
        if missing:
            problems.append(f"arm {name}: unknown features {missing[:5]}")
        absent = sorted(set(VAL_ARM_KEYS) - set(arm))
        if absent:
            problems.append(f"arm {name}: missing keys {absent}")
        other = arm.get("best_iteration_other_seeds")
        if (
            not isinstance(other, dict)
            or set(other) != set(VAL_OTHER_SEEDS)
            or any(not isinstance(v, int) or isinstance(v, bool) for v in other.values())
        ):
            problems.append(
                f"arm {name}: best_iteration_other_seeds must have exactly the keys "
                f"{list(VAL_OTHER_SEEDS)} with int values"
            )
        rounds = arm.get("num_boost_round")
        if not isinstance(rounds, int) or rounds <= 0 or rounds != round(
            int(arm.get("best_iteration_s42", 0)) * gb.REFIT_FACTOR
        ):
            problems.append(f"arm {name}: num_boost_round does not follow the rounds rule")
    ref = binding.get("reference", {})
    if ref.get("arm") != VAL_REFERENCE or list(ref.get("features", [])) != list(hand):
        problems.append("reference arm or its feature list differ from the hand registry")
    per_seed = ref.get("per_seed", {})
    if sorted(per_seed) != sorted(f"s{s}" for s in af.PREREG_SEEDS) or any(
        any(v.get(k) is None for k in VAL_REFERENCE_KEYS) for v in per_seed.values()
    ):
        problems.append("reference per_seed values are incomplete")
    if ref.get("source_log"):
        log = root / ref["source_log"]
        if not log.is_file() or af.file_sha256(log) != ref.get("source_log_sha256"):
            problems.append(f"reference log {ref['source_log']} is missing or differs")
    return problems


def restrict(datas: Mapping[str, ds.WindowData], keep: Sequence[str], every: Sequence[str]):
    drop = set(every) - set(keep)
    return {
        n: d._replace(frame=d.frame.select([c for c in d.frame.columns if c not in drop]))
        for n, d in datas.items()
    }


def build_val_data(
    events: pl.DataFrame,
    columns: Sequence[str],
    cache_dir: str | Path,
    use_cache: bool = True,
    cfg: cd.CandidateConfig = DEFAULT_CONFIG,
    hand: Sequence[ft.FeatureSpec] | None = None,
    generated: Mapping[str, Sequence[ft.FeatureSpec]] | None = None,
    labels: bool = True,
) -> tuple[ds.WindowData, dict] | dict:
    w = sp.get_window(VAL_WINDOW)
    ds.assert_events_before(events, sp.TEST_TS if labels else w.T)
    hand = af.hand_specs() if hand is None else tuple(hand)
    if generated is None:
        generated = {g: af.generated_specs(g) for g in af.GENERATED_GROUPS}
    hist = history(events, w.T)
    sp.assert_no_future(hist, w.T)
    cands, cand_hit = ds.get_or_build_candidates(hist, w.T, cfg, use_cache, cache_dir)
    wanted = set(columns)
    hand_names = {s.name for s in hand}
    plan = [(af.HAND, hand, [c for c in columns if c in hand_names])]
    for g, specs in generated.items():
        plan.append((g, tuple(specs), [s.name for s in specs if s.name in wanted]))
    if {c for _, _, cols in plan for c in cols} != wanted:
        raise KeyError("some requested columns belong to no part")
    parts, files = {}, {}
    for part, specs, cols in plan:
        if not cols:
            continue
        info = af.build_part(hist, w.T, cands, specs, cfg, use_cache, cache_dir)
        parts[part] = {k: info[k] for k in ("file", "specs", "hit", "seconds")}
        files[part] = (Path(cache_dir) / info["file"], cols)
    if not labels:
        return {"candidates_hit": bool(cand_hit), "parts": parts, "n_hist_users": hist["uid"].n_unique()}
    built = af.build_base(events, hist, w, cands)
    if built["n_hist_users"] != hist["uid"].n_unique():
        raise ValueError("history user count mismatch")
    base = built["base"]
    keys = base.select(ds.PAIR)
    feats = []
    for part, (path, cols) in files.items():
        table = pl.scan_parquet(path).select(*ds.PAIR, *cols).collect()
        if not ds._same_keys(table, keys):
            raise ValueError(f"{part} rows of val do not match the candidate rows")
        feats.append(table.drop(ds.PAIR))
    merged = pl.concat(feats, how="horizontal")
    frame = pl.concat(
        [
            base.drop(af.LABEL_COLUMNS),
            merged.select(pl.col(c).cast(pl.Float32) for c in columns),
            base.select(af.LABEL_COLUMNS),
        ],
        how="horizontal",
    )
    npos = built["npos"]
    data = ds.WindowData(
        window=w,
        frame=frame,
        n_pos={lb: npos.filter(pl.col("label") == lb).drop("label") for lb in cd.LABELS},
        n_hist_users=int(built["n_hist_users"]),
        pairs=built["pairs"],
        features=tuple(columns),
        timings={},
        cache={"candidates": bool(cand_hit), "features": all(p["hit"] for p in parts.values())},
    )
    return data, parts


def prepare_val_parts(args, columns: Sequence[str]) -> dict:
    uids = ds.sample_users(args.data, args.users)
    events, _ = af.load_selection_events(args.data, uids)
    out = build_val_data(events, columns, Path(args.cache_dir), labels=False)
    del events
    gc.collect()
    return out


def load_val_data(args, columns: Sequence[str]) -> tuple[ds.WindowData, dict, dict]:
    uids = ds.sample_users(args.data, args.users)
    events, info = ds.load_protocol_events("outer", args.data, uids)
    data, parts = build_val_data(events, columns, Path(args.cache_dir))
    del events
    gc.collect()
    return data, info, parts


def parts_files(parts: Mapping[str, Mapping]) -> dict[str, str]:
    return {part: info["file"] for part, info in parts.items()}


def read_inner_log(prereg_path: Path, body: Mapping) -> dict:
    name = body.get("evidence", {}).get("file")
    if not name:
        return {}
    try:
        log = json.loads((prereg_path.parent / name).read_text())
    except (OSError, ValueError):
        return {}
    return log if isinstance(log, dict) else {}


def inner_log_score(log: Mapping, arm: str, seed: int) -> float | None:
    try:
        value = log["arms"][INNER_LOG_ARMS[arm]]["fits"][f"s{seed}"][f"best_{gb.STOP_METRIC}"]
    except (KeyError, TypeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def val_arm_metrics(
    data: ds.WindowData, key: str, scores, probs: Mapping[str, np.ndarray]
) -> tuple[dict, dict]:
    name = data.window.name
    frame = data.frame.select("uid", "item_id", "novel", *cd.LABELS).with_columns(
        pl.Series(key, np.asarray(scores, dtype=np.float64)),
        *(pl.Series(f"p:{lb}:{key}", np.asarray(probs[lb], dtype=np.float64)) for lb in cd.LABELS),
    )
    grid = {name: {lb: {sl: {} for sl in SLICES} for lb in LABELS}}
    bl._evaluate_keys({name: frame}, {name: data.n_pos}, [key], name, grid)
    metrics = {lb: {sl: grid[name][lb][sl][key] for sl in SLICES} for lb in LABELS}
    all_users = {
        lb: {sl: bl.all_users_metrics(metrics[lb][sl], data.n_hist_users) for sl in SLICES}
        for lb in LABELS
    }
    return metrics, all_users


def inner_fit_arm(
    datas: Mapping[str, ds.WindowData],
    params: Mapping[str, object],
    features: Sequence[str],
    seed: int,
) -> dict:
    fold = af.confirm_fold()
    sel = fold.valid
    cfg = af.confirm_config(params, seed)
    features = list(features)
    t0 = time.perf_counter()
    train, dropped = af.confirm_train(lambda n: datas[n].frame, features, cfg)
    n_pos = ds.group_n_pos(datas, [sel], cfg.label)
    inner = gb.fit(train, datas[sel].frame, cfg, n_pos, features)
    del train
    return {
        "seed": seed,
        "best_iteration": inner.best_iteration,
        "best_score": inner.best_score,
        "scores": gb.predict(inner, datas[sel].frame),
        "train_rows": inner.train_rows,
        "dropped_zero_positive_groups": dropped,
        "seconds": time.perf_counter() - t0,
    }


def refit_arm(
    datas: Mapping[str, ds.WindowData],
    params: Mapping[str, object],
    features: Sequence[str],
    seed: int,
    rounds: int,
    inner: Mapping,
    train_windows: Sequence[str] | None = None,
    target_window: str = VAL_WINDOW,
) -> tuple[np.ndarray, dict, dict]:
    fold = af.confirm_fold()
    sel = fold.valid
    cfg = af.confirm_config(params, seed)
    features = list(features)
    t0 = time.perf_counter()
    cols = [*ds.KEYS, *features, cfg.label]
    windows = (*fold.train, sel) if train_windows is None else tuple(train_windows)
    full = ds.stack_frames([datas[w].frame.select(cols) for w in windows])
    refit = replace(cfg, num_boost_round=rounds, early_stopping_rounds=0)
    final = gb.fit(full, None, refit, None, features)
    del full
    target = datas[target_window]
    scores = gb.predict(final, target.frame)
    probs = {
        lb: gb.to_probability(cfg, lb, target.frame, scores, datas[sel].frame, inner["scores"])[0]
        for lb in cd.LABELS
    }
    info = {
        "seed": seed,
        "inner_best_iteration": inner["best_iteration"],
        f"inner_best_{gb.STOP_METRIC}": inner["best_score"],
        "num_boost_round": rounds,
        "inner_train_rows": inner["train_rows"],
        "refit_train_rows": final.train_rows,
        "dropped_zero_positive_groups": inner["dropped_zero_positive_groups"],
        "inner_seconds": inner["seconds"],
        "refit_seconds": time.perf_counter() - t0,
    }
    return scores, probs, info


def inner_gate(
    inner: Mapping[str, Mapping[int, Mapping]],
    binding: Mapping,
    seeds: Sequence[int],
    inner_log: Mapping | None = None,
):
    inner_log = {} if inner_log is None else inner_log
    problems: list[str] = []
    report: dict = {}
    ref = binding["reference"]["per_seed"]
    report[VAL_REFERENCE] = {}
    for s in seeds:
        got, want = inner[VAL_REFERENCE][s], ref[f"s{s}"]
        rounds = round(got["best_iteration"] * gb.REFIT_FACTOR)
        row = {
            "best_iteration": got["best_iteration"],
            "logged_best_iteration": want["best_iteration"],
            "num_boost_round": rounds,
            "logged_num_boost_round": want["num_boost_round"],
            "inner_ndcg@10": got["best_score"],
            "logged_inner_ndcg@10": want["inner_train_280_ndcg@10"],
        }
        row["match"] = bool(
            row["best_iteration"] == row["logged_best_iteration"]
            and rounds == row["logged_num_boost_round"]
            and round(row["inner_ndcg@10"], VAL_DIGITS) == round(row["logged_inner_ndcg@10"], VAL_DIGITS)
        )
        if not row["match"]:
            problems.append(f"{VAL_REFERENCE} s{s}: inner fit differs from the pre-registered values {row}")
        report[VAL_REFERENCE][f"s{s}"] = row
    for arm in af.PREREG_ARMS:
        spec = binding["arms"][arm]
        expected = {f"s{gb.SEED}": spec["best_iteration_s42"], **spec["best_iteration_other_seeds"]}
        report[arm] = {}
        for s in seeds:
            want = expected.get(f"s{s}")
            got = inner[arm][s]["best_iteration"]
            row = {"best_iteration": got, "prereg_best_iteration": want}
            if want is None:
                row.update({"match": None, "note": "no pre-registered reference for this seed, not checked"})
                report[arm][f"s{s}"] = row
                continue
            score = inner[arm][s]["best_score"]
            pinned = inner_log_score(inner_log, arm, s)
            row["inner_ndcg@10"] = score
            row["pinned_inner_ndcg@10"] = pinned
            iteration_ok = got == want
            score_ok = pinned is not None and round(score, VAL_DIGITS) == round(pinned, VAL_DIGITS)
            row["match"] = bool(iteration_ok and score_ok)
            if not iteration_ok:
                problems.append(f"{arm} s{s}: inner best_iteration {got} differs from the pre-registered {want}")
            if not score_ok:
                problems.append(
                    f"{arm} s{s}: inner ndcg@10 {score} differs from the pinned inner log value {pinned} "
                    f"at {VAL_DIGITS} decimals"
                )
            report[arm][f"s{s}"] = row
    return problems, report


def reference_check(measured: Mapping[int, float], reference: Mapping, seeds: Sequence[int]) -> dict:
    per_seed = {}
    for s in seeds:
        logged = reference["per_seed"][f"s{s}"]["val_ndcg@10"]
        mine = measured[s]
        per_seed[f"s{s}"] = {
            "ndcg@10": mine,
            "logged_ndcg@10": logged,
            "delta": mine - logged,
            "match": round(mine, VAL_DIGITS) == round(logged, VAL_DIGITS),
        }
    return {
        "rule": VAL_RULES["reference"],
        "digits": VAL_DIGITS,
        "per_seed": per_seed,
        "all_match": all(v["match"] for v in per_seed.values()),
    }


def logged_metric_diffs(measured: Mapping[str, Mapping], logged: Mapping) -> dict:
    out = {}
    for key, mine in measured.items():
        diffs = []
        for lb, by_slice in mine.items():
            for sl, values in by_slice.items():
                old = logged.get(lb, {}).get(sl, {}).get(key, {})
                diffs.extend(
                    abs(v - old[m])
                    for m, v in values.items()
                    if isinstance(v, (int, float)) and isinstance(old.get(m), (int, float))
                )
        out[key] = max(diffs) if diffs else None
    return out


def val_bootstrap(means: Mapping[str, Mapping[str, np.ndarray]], binding: Mapping) -> dict:
    hyp = binding["hypotheses"]
    levels = {
        "G127": hyp["H1"]["ci_level"],
        "G53": hyp["secondary"]["ci_level"],
        "U": hyp["H2"]["ci_level"],
    }
    boot = binding["bootstrap"]
    return {
        sl: {
            f"{a} vs {VAL_REFERENCE}": af.paired_bootstrap(
                means[a][sl], means[VAL_REFERENCE][sl], boot["n_boot"], boot["seed"], levels[a]
            )
            for a in af.PREREG_ARMS
        }
        for sl in BOOT_SLICES
    }


def val_decision(boot_all: Mapping[str, Mapping], binding: Mapping) -> dict:
    hyp = binding["hypotheses"]
    margin = hyp["H1"]["margin"]

    def view(b):
        return {k: b[k] for k in ("ci_level", "mean_a", "mean_b", "mean_delta", "ci_low", "ci_high")}

    g, u, g53 = (boot_all[f"{a} vs {VAL_REFERENCE}"] for a in ("G127", "U", "G53"))
    h1_holds = g["ci_low"] is not None and g["ci_low"] > -margin
    h2_holds = u["ci_low"] is not None and u["ci_low"] > 0
    return {
        "H1": {
            "arm": "G127",
            "kind": "non-inferiority",
            "margin": margin,
            "threshold": -margin,
            **view(g),
            "holds": bool(h1_holds),
            "verdict": "non-inferior within the margin" if h1_holds else "non-inferiority not shown",
        },
        "H2": {
            "arm": "U",
            "kind": "superiority",
            "threshold": 0.0,
            **view(u),
            "holds": bool(h2_holds),
            "verdict": "superior" if h2_holds else "superiority not shown",
        },
        "secondary": {
            "G53": {**view(g53), "ci_excludes_zero": g53["ci_excludes_zero"]},
            "U": {**view(u), "ci_excludes_zero": u["ci_excludes_zero"]},
        },
        "not_val_items": "H3, H4 and H5 are not val-time items of the pre-registration",
    }


def val_record_base(
    seeds: Sequence[int],
    stem: str,
    prereg_path: Path,
    body: Mapping,
    verify: Mapping,
    params: Mapping,
    params_source: Mapping,
    loaded: Sequence[str],
    data_info: Mapping,
    guard: Path,
) -> dict:
    return {
        "seed": gb.SEED,
        "seeds": list(seeds),
        "git_commit": gb.GIT_COMMIT,
        "stage": "val",
        "protocol": "outer",
        "selection_window": af.confirm_fold().valid,
        "target_split": VAL_WINDOW,
        "loaded_windows": list(loaded),
        "config": {
            "version": VAL_VERSION,
            "prereg": {
                "file": prereg_path.name,
                "file_sha256": af.file_sha256(prereg_path),
                "decision_digest": body["decision_digest"],
                "metrics_digest": body["metrics_digest"],
                "verify": dict(verify),
            },
            "params": dict(params),
            "params_source": dict(params_source),
            "fit_config": af.confirm_config(params, seeds[0]).config_dict(),
            "num_boost_round": {a: body["binding"]["arms"][a]["num_boost_round"] for a in af.PREREG_ARMS},
            "rules": VAL_RULES,
            "guard": {"file": guard.name, "stem": stem},
        },
        "data": dict(data_info),
        "versions": gb.library_versions(),
    }


def stage_val(args) -> int:
    out_dir, runs_dir = Path(args.out_dir), Path(args.runs_dir)
    hits = val_guard_hits(out_dir, runs_dir)
    if hits:
        print(f"VAL GUARD: refusing to run, found {hits}", file=sys.stderr)
        return VAL_GUARD_EXIT_CODE
    params, params_source = load_params(args.params_json)
    hand = list(gb.resolve_features(gb.GBDTConfig(params=params), af.hand_specs()))
    known = [s.name for s in (*af.hand_specs(), *af.generated_specs())]
    try:
        prereg_path = resolve_prereg(runs_dir, args.prereg)
        verify = af.verify_prereg(prereg_path)
        body = json.loads(prereg_path.read_text())
        problems = [*verify["diff"], *check_binding(
            body.get("binding", {}), hand, known, params_source["sha256"]
        )]
        dirs = {out_dir.resolve(), runs_dir.resolve(), prereg_path.parent.resolve()}
        if len(dirs) != 1:
            problems.append(
                "out-dir, runs-dir and the prereg folder must be the same directory, got "
                f"{sorted(str(d) for d in dirs)}"
            )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        problems = [f"{type(exc).__name__}: {exc}"]
    if problems:
        print("VAL PREREG: verification failed, nothing was loaded", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return VAL_PREREG_EXIT_CODE
    binding = body["binding"]
    union = list(
        dict.fromkeys([*hand, *(c for a in af.PREREG_ARMS for c in binding["arms"][a]["features"])])
    )
    if args.dry_run:
        man = af.read_manifest(args.users, args.cache_dir)
        have = {c for names in man["columns"].values() for c in names}
        absent = sorted(set(union) - have)
        if absent:
            print(f"VAL DRY RUN: columns missing from the manifest: {absent[:10]}", file=sys.stderr)
            return VAL_PREREG_EXIT_CODE
        print(f"val dry run ok: {prereg_path.name} verified, {len(union)} columns, no frame loaded")
        return 0
    now = datetime.now()
    stem = f"{now:{STEM_FORMAT}}"
    state: dict = {"stage": "setup", "completed": [], "info": {"stem": stem}}
    dog = Watchdog(args.max_rss_gb * GIB, state, out_dir / f"{stem}-autofeat-val-partial.json")
    dog.start()
    try:
        return run_val(
            args, now, stem, prereg_path, body, verify, hand, union, params, params_source, state, dog
        )
    finally:
        dog.stop()


def per_user_mismatch(checks: Mapping[str, float]) -> list[str]:
    return sorted(k for k, v in checks.items() if v > VAL_PER_USER_TOL)


def run_val(
    args,
    now: datetime,
    stem: str,
    prereg_path: Path,
    body: Mapping,
    verify: Mapping,
    hand: Sequence[str],
    union: Sequence[str],
    params: Mapping,
    params_source: Mapping,
    state: dict,
    dog: Watchdog,
) -> int:
    start = time.perf_counter()
    runtime: dict[str, float] = {}
    out_dir, cache_dir = Path(args.out_dir), Path(args.cache_dir)
    binding = body["binding"]
    reference = binding["reference"]
    seeds = list(binding["val_seeds"])
    arm_features = {a: list(binding["arms"][a]["features"]) for a in af.PREREG_ARMS}
    rounds = {a: int(binding["arms"][a]["num_boost_round"]) for a in af.PREREG_ARMS}
    all_features = list(union)
    state["stage"] = "load train windows"
    t0 = time.perf_counter()
    loaded: list[str] = []
    datas: dict[str, ds.WindowData] = {}
    for w in af.confirm_window_names():
        datas[w] = af.confirm_window_data(w, union, args.users, cache_dir)
        loaded.append(w)
        print(f"  loaded {w} {datas[w].frame.shape} rss {process_rss() / MIB:.0f} MB", flush=True)
    runtime["load_train_windows"] = time.perf_counter() - t0
    state["stage"] = "val candidates and features (label-free)"
    t0 = time.perf_counter()
    prepared = prepare_val_parts(args, union)
    runtime["prepare_val_parts"] = time.perf_counter() - t0
    print(f"  val candidates and feature parts ready rss {process_rss() / MIB:.0f} MB", flush=True)
    feature_sets = {VAL_REFERENCE: list(hand), **arm_features}
    inner: dict[str, dict[int, dict]] = {}
    t0 = time.perf_counter()
    for arm, feats in feature_sets.items():
        state["stage"] = f"inner fits {arm}"
        sub = restrict(datas, feats, all_features)
        inner[arm] = {}
        for s in seeds:
            inner[arm][s] = inner_fit_arm(sub, params, feats, s)
            print(
                f"  inner {arm}:s{s} best_iter {inner[arm][s]['best_iteration']} "
                f"ndcg@10 {inner[arm][s]['best_score']:.6f}",
                flush=True,
            )
        del sub
        gc.collect()
    runtime["inner_fits"] = time.perf_counter() - t0
    gate_problems, gate_report = inner_gate(inner, binding, seeds, read_inner_log(prereg_path, body))
    if gate_problems:
        print("VAL INNER GATE: label-free inner fits differ from the pre-registration, val untouched", file=sys.stderr)
        for p in gate_problems:
            print(f"  {p}", file=sys.stderr)
        return VAL_PREREG_EXIT_CODE
    print("inner gate passed: E1a and arm inner fits equal the pre-registered values", flush=True)
    state["stage"] = "guard"
    try:
        guard = claim_val_guard(
            out_dir,
            {
                "created_at": now.isoformat(timespec="seconds"),
                "stem": stem,
                "pid": os.getpid(),
                "prereg": prereg_path.name,
                "decision_digest": body["decision_digest"],
                "note": "created before any val label is read; remove manually only after a "
                "documented decision",
            },
        )
    except ValGuardError as exc:
        print(f"VAL GUARD: {exc}", file=sys.stderr)
        return VAL_GUARD_EXIT_CODE
    print(f"val guard written: {guard}", flush=True)
    state["stage"] = "load val"
    t0 = time.perf_counter()
    datas[VAL_WINDOW], data_info, val_parts = load_val_data(args, union)
    loaded.append(VAL_WINDOW)
    parts_match = parts_files(val_parts) == parts_files(prepared["parts"])
    if not parts_match:
        print(
            "VAL WARNING: feature part files after the guard differ from the label-free preparation: "
            f"{parts_files(val_parts)} vs {parts_files(prepared['parts'])}",
            file=sys.stderr,
        )
    runtime["load_val"] = time.perf_counter() - t0
    val = datas[VAL_WINDOW]
    print(
        f"  built {VAL_WINDOW} {val.frame.shape} users {val.n_hist_users} rss {process_rss() / MIB:.0f} MB",
        flush=True,
    )
    base = val_record_base(
        seeds, stem, prereg_path, body, verify, params, params_source, loaded, data_info, guard
    )
    base["inner_gate"] = {
        "passed": True,
        "report": gate_report,
        "prepared_parts": prepared,
        "post_guard_parts_match": parts_match,
    }
    arms: dict[str, dict] = {}
    metrics: dict[str, dict] = {}
    all_users: dict[str, dict] = {}
    vectors: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    uids: dict[str, np.ndarray] = {}
    checks: dict[str, float] = {}

    def record_fit(arm: str, seed: int, scores, probs, info: dict) -> None:
        key = f"{arm}:s{seed}"
        t = time.perf_counter()
        metrics[key], all_users[key] = val_arm_metrics(val, key, scores, probs)
        info["metric_seconds"] = time.perf_counter() - t
        vectors.setdefault(arm, {})[seed] = {}
        for sl in BOOT_SLICES:
            u, v = af.per_user_ndcg(val.frame, scores, val.n_pos[af.CONFIRM_LABEL], af.CONFIRM_LABEL, sl)
            if sl in uids and not np.array_equal(uids[sl], u):
                raise ValueError("per-user vectors are not aligned")
            uids[sl] = u
            vectors[arm][seed][sl] = v
            checks[f"{sl}/{key}"] = abs(float(v.mean()) - metrics[key][af.CONFIRM_LABEL][sl]["ndcg@10"])
        arms.setdefault(arm, {"fits": {}})["fits"][f"s{seed}"] = info
        print(
            f"  {key:12} ndcg@10 {metrics[key]['y_org']['all']['ndcg@10']:.6f} "
            f"inner_best_iter {info['inner_best_iteration']} rounds {info['num_boost_round']}",
            flush=True,
        )

    def finish_without_verdict(status: str, note: str, extra: Mapping, code: int) -> int:
        state["stage"] = "write discrepancy"
        runtime["total"] = time.perf_counter() - start
        record = {
            **base,
            "status": status,
            "verdict": None,
            "verdict_note": note,
            **extra,
            "arms": arms,
            "per_user_check": checks,
            "metrics": {VAL_WINDOW: grid_by_label(metrics)},
            "metrics_all_users": {VAL_WINDOW: grid_by_label(all_users)},
            "runtime_seconds": runtime,
            "resources": {"peak_rss_mb_lifetime": gb.peak_rss_mb(), "peak_rss_mb_sampled": dog.peak_bytes / MIB},
        }
        path, _ = gb.write_run_log(record, "autofeat-val", out_dir, now)
        print(f"{status.upper()} on val, no verdict. log {path}", file=sys.stderr)
        return code

    state["stage"] = f"fit {VAL_REFERENCE}"
    t0 = time.perf_counter()
    ref_datas = restrict(datas, hand, all_features)
    measured: dict[int, float] = {}
    for s in seeds:
        n_rounds = round(inner[VAL_REFERENCE][s]["best_iteration"] * gb.REFIT_FACTOR)
        scores, probs, info = refit_arm(ref_datas, params, hand, s, n_rounds, inner[VAL_REFERENCE][s])
        record_fit(VAL_REFERENCE, s, scores, probs, info)
        measured[s] = metrics[f"{VAL_REFERENCE}:s{s}"]["y_org"]["all"]["ndcg@10"]
        del scores, probs
        gc.collect()
    del ref_datas
    runtime["fit_reference"] = time.perf_counter() - t0
    state["completed"].append(VAL_REFERENCE)
    r0 = reference_check(measured, reference, seeds)
    for s in seeds:
        r0["per_seed"][f"s{s}"].update(
            {
                "inner_best_iteration": arms[VAL_REFERENCE]["fits"][f"s{s}"]["inner_best_iteration"],
                "logged_best_iteration": reference["per_seed"][f"s{s}"]["best_iteration"],
                "num_boost_round": arms[VAL_REFERENCE]["fits"][f"s{s}"]["num_boost_round"],
                "logged_num_boost_round": reference["per_seed"][f"s{s}"]["num_boost_round"],
            }
        )
    e1_log = reference.get("source_log")
    if e1_log:
        logged = json.loads((af.PROJECT_ROOT / e1_log).read_text())["metrics"]["outer"][VAL_WINDOW]
        keys = [f"{VAL_REFERENCE}:s{s}" for s in seeds]
        r0["max_abs_diff_all_logged_metrics"] = logged_metric_diffs(
            {k: metrics[k] for k in keys}, logged
        )
    arms[VAL_REFERENCE].update(
        {
            "n_features": len(hand),
            "features_sha256": af.list_sha256(hand),
            "seeds": seeds,
            "rounds_rule": reference.get("rounds_rule"),
        }
    )
    if not r0["all_match"]:
        for k, v in r0["per_seed"].items():
            print(f"  {k}: {v['ndcg@10']:.6f} vs logged {v['logged_ndcg@10']:.6f} match {v['match']}", file=sys.stderr)
        return finish_without_verdict(
            "r0_mismatch",
            "E1a did not reproduce the logged val values to "
            f"{VAL_DIGITS} decimals; the other arms were not fitted and no comparison is made",
            {"r0_check": r0},
            R0_EXIT_CODE,
        )
    print(f"E1a reproduced on val for seeds {seeds}", flush=True)
    for arm in af.PREREG_ARMS:
        state["stage"] = f"fit {arm}"
        feats = arm_features[arm]
        t0 = time.perf_counter()
        sub = restrict(datas, feats, all_features)
        for s in seeds:
            scores, probs, info = refit_arm(sub, params, feats, s, rounds[arm], inner[arm][s])
            expected = gate_report[arm][f"s{s}"]["prereg_best_iteration"]
            info["prereg_inner_best_iteration"] = expected
            info["inner_matches_prereg"] = gate_report[arm][f"s{s}"]["match"]
            record_fit(arm, s, scores, probs, info)
            del scores, probs
            gc.collect()
        del sub
        arms[arm].update(
            {
                "n_features": len(feats),
                "features_sha256": af.list_sha256(feats),
                "seeds": seeds,
                "num_boost_round": rounds[arm],
                "role": binding["arms"][arm]["role"],
            }
        )
        runtime[f"fit_{arm}"] = time.perf_counter() - t0
        state["completed"].append(arm)
    bad = per_user_mismatch(checks)
    if bad:
        return finish_without_verdict(
            "per_user_mismatch",
            f"per-user ndcg@10 means differ from the metric by more than {VAL_PER_USER_TOL:g} for {bad[:5]}",
            {"r0_check": r0, "per_user_mismatch": bad},
            R0_EXIT_CODE,
        )
    state["stage"] = "bootstrap"
    means = {a: {sl: np.mean([vectors[a][s][sl] for s in seeds], axis=0) for sl in BOOT_SLICES} for a in arms}
    boot = val_bootstrap(means, binding)
    decision = val_decision(boot["all"], binding)
    keys = list(metrics)
    stats = {a: gb.seed_stats({VAL_WINDOW: grid_by_label(metrics, keys)}, [f"{a}:s{s}" for s in seeds])[VAL_WINDOW] for a in arms}
    for a, rec in arms.items():
        rec["summary"] = summarize(metrics, [f"{a}:s{s}" for s in seeds])
    runtime["total"] = time.perf_counter() - start
    record = {
        **base,
        "status": "completed",
        "r0_check": r0,
        "arms": arms,
        "metrics_layout": "metrics[split][label][slice][arm:seed] -> metric dict",
        "metrics": {VAL_WINDOW: grid_by_label(metrics, keys)},
        "metrics_all_users": {VAL_WINDOW: grid_by_label(all_users, keys)},
        "seed_stats": stats,
        "bootstrap": {
            "rule": VAL_RULES["bootstrap"],
            "label": af.CONFIRM_LABEL,
            "metric": af.PREREG_METRIC,
            "slices": boot,
        },
        "per_user_check": checks,
        "decision": decision,
        "runtime_seconds": runtime,
        "resources": {
            "peak_rss_mb_lifetime": gb.peak_rss_mb(),
            "peak_rss_mb_sampled": dog.peak_bytes / MIB,
        },
    }
    path, digest = gb.write_run_log(record, "autofeat-val", out_dir, now)
    print_val(record)
    print(f"val log: {path}")
    print(f"metrics_digest: {digest}")
    return 0


def grid_by_label(table: Mapping[str, Mapping], keys: Sequence[str] | None = None) -> dict:
    keys = list(table) if keys is None else keys
    return {lb: {sl: {k: table[k][lb][sl] for k in keys} for sl in SLICES} for lb in LABELS}


def print_val(record: Mapping) -> None:
    print("val (y_org): arm, n_feat, ndcg@10 mean+-std over seeds, recall@10, recall_cap@10, novel ndcg@10")
    for a, r in record["arms"].items():
        s = r["summary"]
        print(
            f"  {a:6} {r['n_features']:4d} {s['y_org/all/ndcg@10']['mean']:.4f}+-"
            f"{s['y_org/all/ndcg@10']['std'] or 0.0:.4f} {s['y_org/all/recall@10']['mean']:.4f} "
            f"{s['y_org/all/recall_cap@10']['mean']:.4f} {s['y_org/novel/ndcg@10']['mean']:.4f}"
        )
    print("bootstrap ndcg@10 (y_org, seed-averaged per user):")
    for sl, pairs in record["bootstrap"]["slices"].items():
        for name, r in pairs.items():
            print(
                f"  {sl:6} {name:14} {r['mean_delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
                f"level {r['ci_level']}"
            )
    d = record["decision"]
    print(f"H1 holds {d['H1']['holds']} ({d['H1']['verdict']}) lower bound {d['H1']['ci_low']:+.4f} vs {d['H1']['threshold']:+.4f}")
    print(f"H2 holds {d['H2']['holds']} ({d['H2']['verdict']}) lower bound {d['H2']['ci_low']:+.4f}")


STAGE_FLAGS = {
    "dry_run": ("val",),
    "prereg": ("val",),
    "skip_r1": ("confirm",),
    "r1_only": ("confirm",),
    "run_stem": ("confirm",),
    "screen_log": ("select",),
    "skip_greedy": ("select",),
    "greedy_from": ("select",),
    "greedy_budget": ("select",),
    "no_cache": ("build",),
    "min_free_gb": ("build",),
    "batch_size": ("screen",),
    "workers": ("screen",),
    "max_rss_gb": ("confirm", "val"),
    "name": ("build", "screen", "select"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--users", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--cache-dir", default=str(af.CACHE_DIR))
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--params-json", default=str(DEFAULT_PARAMS_JSON))
    parser.add_argument("--name", default=None)
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    parser.add_argument("--batch-size", type=int, default=af.SCREEN_BATCH)
    parser.add_argument("--workers", type=int, default=af.SCREEN_WORKERS)
    parser.add_argument("--screen-log", default=None)
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--greedy-from", default=None)
    parser.add_argument("--greedy-budget", type=float, default=af.GREEDY_BUDGET)
    parser.add_argument("--skip-r1", action="store_true")
    parser.add_argument("--r1-only", action="store_true")
    parser.add_argument("--run-stem", default=None)
    parser.add_argument("--max-rss-gb", type=float, default=MAX_RSS_GB)
    parser.add_argument("--prereg", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for dest, stages in STAGE_FLAGS.items():
        if args.stage not in stages and getattr(args, dest) != parser.get_default(dest):
            parser.error(f"--{dest.replace('_', '-')} is not applicable to stage {args.stage}")
    if args.users < 0:
        parser.error("users must be >= 0")
    if args.batch_size < 1 or args.workers < 1:
        parser.error("batch-size and workers must be >= 1")
    if args.name is None:
        args.name = f"autofeat-{args.stage}"
    if args.stage == "screen":
        return stage_screen(args)
    if args.stage == "select":
        return stage_select(args)
    if args.stage == "confirm":
        if args.max_rss_gb <= 0:
            parser.error("max-rss-gb must be > 0")
        return stage_confirm(args)
    if args.stage == "val":
        if args.max_rss_gb <= 0:
            parser.error("max-rss-gb must be > 0")
        if args.users != 0:
            parser.error("the val stage runs on the full sample only (users 0)")
        return stage_val(args)
    try:
        return stage_build(args)
    except DiskSpaceError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return DISK_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
