import argparse
import gc
import json
import math
import os
import platform
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import psutil

from recsys_tbank import autofeat as af
from recsys_tbank import dataset as ds
from recsys_tbank import features as ft
from recsys_tbank import gbdt as gb
from recsys_tbank.candidates import DEFAULT_CONFIG, SOURCES, build_candidates
from recsys_tbank.split import DATA_PATH, VAL_TS, get_window, history

SEED = 42
WINDOW = "val"
SIZES = (250, 500, 1000, 10000)
FIT_SIZES = (250, 500, 1000)
SCALE_USERS = 10000
DEV_USERS = 1000
MAX_RSS_GB = 10.0
WATCH_INTERVAL = 2.0
ABORT_EXIT_CODE = 3
GIB = 1024**3
MIB = 1024**2
SPECS_HAND = "hand"
SPECS_GENERATED = "generated"
SPECS_PREREG = "prereg"
PREREG_GLOB = "*-autofeat-prereg.json"
ROOT = DATA_PATH.parents[2]
SOURCE_PATH = ROOT / "data" / "yambda" / "flat" / "50m" / "multi_event.parquet"
SCALE_PATH = ROOT / "data" / "scale" / "events_10000u.parquet"


def pool_uids(path: str | Path) -> np.ndarray:
    return (
        pl.scan_parquet(path)
        .filter(pl.col("timestamp") < VAL_TS)
        .select(pl.col("uid").cast(pl.UInt32))
        .unique()
        .sort("uid")
        .collect()["uid"]
        .to_numpy()
    )


def shuffled_order(uids, seed: int = SEED) -> np.ndarray:
    ordered = np.unique(np.asarray(uids))
    return np.random.default_rng(seed).permutation(ordered)


def nested_sample(uids, n: int, seed: int = SEED) -> np.ndarray:
    order = shuffled_order(uids, seed)
    if n <= 0 or n > order.size:
        raise ValueError(f"cannot sample {n} users from a pool of {order.size}")
    return np.sort(order[:n])


def fit_loglog(sizes: Sequence[float], values: Sequence[float]) -> dict | None:
    pairs = [
        (float(s), float(v))
        for s, v in zip(sizes, values)
        if v is not None and s > 0 and v > 0 and math.isfinite(v)
    ]
    if len(pairs) < 2 or len({s for s, _ in pairs}) < 2:
        return None
    x = np.log([s for s, _ in pairs])
    y = np.log([v for _, v in pairs])
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_points": len(pairs),
    }


def fit_linear(sizes: Sequence[float], values: Sequence[float]) -> dict | None:
    pairs = [
        (float(s), float(v))
        for s, v in zip(sizes, values)
        if v is not None and math.isfinite(v)
    ]
    if len({s for s, _ in pairs}) < 2:
        return None
    slope, intercept = np.polyfit([s for s, _ in pairs], [v for _, v in pairs], 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def extrapolate(fit: Mapping, n: float) -> float:
    return float(math.exp(fit["intercept"] + fit["slope"] * math.log(n)))


def flat_measures(result: Mapping) -> dict[str, float]:
    out = {
        k: float(v)
        for k, v in result.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    out.update({f"seconds/{k}": float(v) for k, v in result["seconds"].items()})
    return out


def scaling_fits(
    results: Sequence[Mapping],
    fit_sizes: Sequence[int] = FIT_SIZES,
    target: int = SCALE_USERS,
) -> dict:
    used = [r for r in results if r["n_users"] in fit_sizes]
    measured = {r["n_users"]: flat_measures(r) for r in results}
    names = sorted({k for r in used for k in flat_measures(r)})
    out = {}
    for name in names:
        sizes = [r["n_users"] for r in used if name in flat_measures(r)]
        values = [flat_measures(r)[name] for r in used if name in flat_measures(r)]
        fit = fit_loglog(sizes, values)
        if fit is None:
            continue
        linear = fit_linear(sizes, values)
        out[name] = {
            **fit,
            "linear": linear,
            f"extrapolated_linear_{target}": None
            if linear is None
            else linear["intercept"] + linear["slope"] * target,
            "fit_sizes": sizes,
            f"extrapolated_{target}": extrapolate(fit, target),
            f"measured_{target}": measured.get(target, {}).get(name),
        }
    return out


def exceeds(rss_bytes: float, limit_bytes: float) -> bool:
    return rss_bytes > limit_bytes


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
        clock: Callable[[], float] = time.time,
    ):
        super().__init__(daemon=True)
        self.limit_bytes = limit_bytes
        self.state = state
        self.partial_path = Path(partial_path)
        self.rss_fn = rss_fn
        self.exit_fn = exit_fn
        self.interval = interval
        self.clock = clock
        self.peak_bytes = 0.0
        self.stopper = threading.Event()

    def reset_peak(self) -> None:
        self.peak_bytes = 0.0

    def partial_record(self, rss: float) -> dict:
        return {
            "status": "aborted_rss_limit",
            "stage": self.state.get("stage"),
            "n_users": self.state.get("n_users"),
            "rss_bytes": int(rss),
            "rss_gb": rss / GIB,
            "limit_gb": self.limit_bytes / GIB,
            "timestamp": self.clock(),
            "timestamp_iso": datetime.fromtimestamp(self.clock()).isoformat(
                timespec="seconds"
            ),
            "load_average": list(os.getloadavg()),
            "completed": self.state.get("completed", []),
        }

    def write_partial(self, rss: float) -> None:
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.partial_path.with_name(f".{self.partial_path.name}.tmp")
        tmp.write_text(json.dumps(self.partial_record(rss), indent=2) + "\n")
        os.replace(tmp, self.partial_path)

    def step(self) -> bool:
        rss = self.rss_fn()
        self.peak_bytes = max(self.peak_bytes, rss)
        if not exceeds(rss, self.limit_bytes):
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


def build_scale_sample(
    source: Path, dest: Path, n: int = SCALE_USERS, seed: int = SEED
) -> dict:
    start = time.perf_counter()
    lazy = pl.scan_parquet(source)
    users = lazy.select("uid").unique().sort("uid").collect()
    total = users.height
    dest.parent.mkdir(parents=True, exist_ok=True)
    if total <= n:
        lazy.sink_parquet(dest)
    else:
        chosen = users.sample(n=n, seed=seed)
        lazy.join(chosen.lazy(), on="uid", how="semi").sink_parquet(dest)
    scanned = pl.scan_parquet(dest).select(pl.len(), pl.col("uid").n_unique()).collect()
    return {
        "source": str(source),
        "path": str(dest),
        "users_in_source": total,
        "users_requested": n,
        "whole_source": total <= n,
        "rows": int(scanned.item(0, 0)),
        "users": int(scanned.item(0, 1)),
        "bytes": dest.stat().st_size,
        "seconds": time.perf_counter() - start,
    }


def timed(fn: Callable, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, time.perf_counter() - start


def set_stage(state: dict, n: int, stage: str) -> None:
    state["n_users"] = n
    state["stage"] = stage
    print(f"[{n}] {stage}", flush=True)


def specs_arg(value: str) -> str:
    mode, arm = split_specs(value)
    if mode not in (SPECS_HAND, SPECS_GENERATED, SPECS_PREREG) or (
        (mode == SPECS_PREREG) != (arm is not None)
    ):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not hand, generated or prereg:<arm>"
        )
    if arm == "":
        raise argparse.ArgumentTypeError("prereg needs an arm name, prereg:<arm>")
    return value


def split_specs(value: str) -> tuple[str, str | None]:
    mode, sep, arm = value.partition(":")
    return mode, (arm if sep else None)


def find_prereg(runs_dir: str | Path) -> Path:
    found = sorted(Path(runs_dir).glob(PREREG_GLOB))
    if not found:
        raise FileNotFoundError(f"no {PREREG_GLOB} in {runs_dir}")
    return found[-1]


def prereg_arm(path: str | Path, arm: str) -> dict:
    arms = json.loads(Path(path).read_text())["binding"]["arms"]
    if arm not in arms:
        raise ValueError(f"unknown arm {arm!r}, available: {sorted(arms)}")
    return arms[arm]


def feature_parts(
    mode: str, arm: str | None = None, prereg_path: str | Path | None = None
) -> list[tuple[str, tuple]] | None:
    if mode == SPECS_HAND:
        return None
    if mode == SPECS_GENERATED:
        return [(g, af.generated_specs(g)) for g in af.GENERATED_GROUPS]
    if mode != SPECS_PREREG or arm is None or prereg_path is None:
        raise ValueError(f"cannot resolve specs mode {mode!r}")
    return prereg_parts(arm, prereg_path)


def check_generated_names(arm: str, wanted: set, chosen: set) -> None:
    if not chosen <= wanted:
        raise ValueError(f"arm {arm}: generated names are not a subset of features")
    catalog = {s.name for s in af.generated_specs()}
    unknown = sorted(chosen - catalog)
    if unknown:
        raise ValueError(f"arm {arm}: unknown generated features {unknown[:5]}")


def check_hand_names(arm: str, wanted: set, chosen: set, hand: tuple) -> None:
    lost = sorted(wanted - chosen - {s.name for s in hand})
    if lost:
        raise ValueError(f"arm {arm}: unknown hand features {lost[:5]}")


def chosen_parts(hand: tuple, chosen: set) -> list[tuple[str, tuple]]:
    parts = [(SPECS_HAND, hand)] if hand else []
    for g in af.GENERATED_GROUPS:
        specs = tuple(s for s in af.generated_specs(g) if s.name in chosen)
        if specs:
            parts.append((g, specs))
    return parts


def prereg_parts(arm: str, prereg_path: str | Path) -> list[tuple[str, tuple]]:
    record = prereg_arm(prereg_path, arm)
    wanted = set(record["features"])
    chosen = set(record["generated"])
    check_generated_names(arm, wanted, chosen)
    hand = tuple(s for s in af.hand_specs() if s.name in wanted - chosen)
    check_hand_names(arm, wanted, chosen, hand)
    return chosen_parts(hand, chosen)


def build_parts(
    hist,
    T: int,
    cands: pl.DataFrame,
    cfg,
    parts: Sequence[tuple[str, Sequence]],
    n: int,
    state: dict,
    dog: Watchdog,
) -> tuple[dict[str, float], list[dict]]:
    seconds: dict[str, float] = {"features": 0.0, "features/prep": 0.0}
    infos: list[dict] = []
    with af.generated_registered():
        for part, specs in parts:
            set_stage(state, n, f"features/{part}")
            carried = dog.peak_bytes
            dog.reset_peak()
            stages: dict[str, float] = {}
            (frame, _), took = timed(
                ds.get_or_build_features,
                hist,
                T,
                cands,
                tuple(specs),
                cfg,
                False,
                timings=stages,
            )
            built = process_rss()
            info = {
                "part": part,
                "specs": len(specs),
                "seconds": took,
                "rows": frame.height,
                "columns": frame.width - 2,
                "bytes": frame.estimated_size(),
                "rss_mb_after_build": built / MIB,
            }
            del frame
            gc.collect()
            peak = max(dog.peak_bytes, built)
            dog.peak_bytes = max(carried, peak)
            info["peak_rss_mb_sampled"] = peak / MIB
            info["rss_mb_after_discard"] = process_rss() / MIB
            info["stages"] = stages
            infos.append(info)
            seconds["features"] += took
            for k, v in stages.items():
                key = "features/prep" if k == "prep" else f"features/{k}"
                seconds[key] = seconds.get(key, 0.0) + v
    return seconds, infos


def candidate_sources(
    hist, T: int, cfg, n: int, state: dict, breakdown: bool
) -> dict[str, float]:
    sources: dict[str, float] = {}
    if breakdown:
        for s in SOURCES:
            set_stage(state, n, f"candidates/{s}")
            _, sources[s] = timed(build_candidates, hist, T, cfg, (s,))
    return sources


def hand_feature_stats(
    hist, T: int, cands: pl.DataFrame, cfg, specs, n: int, state: dict
) -> tuple[dict, dict[str, float], float, list[dict]]:
    groups: dict[str, float] = {}
    set_stage(state, n, "features")
    (feats, _), took = timed(
        ds.get_or_build_features,
        hist,
        T,
        cands,
        specs,
        cfg,
        False,
        timings=groups,
    )
    feature_stats = {
        "rows": feats.height,
        "columns": feats.width - 2,
        "bytes": feats.estimated_size(),
    }
    del feats
    return feature_stats, {f"features/{k}": v for k, v in groups.items()}, took, []


def part_feature_stats(
    hist,
    T: int,
    cands: pl.DataFrame,
    cfg,
    parts: Sequence[tuple[str, Sequence]],
    n: int,
    state: dict,
    dog: Watchdog,
) -> tuple[dict, dict[str, float], float, list[dict]]:
    part_seconds, part_infos = build_parts(hist, T, cands, cfg, parts, n, state, dog)
    took = part_seconds.pop("features")
    feature_stats = {
        "rows": part_infos[0]["rows"] if part_infos else cands.height,
        "columns": sum(i["columns"] for i in part_infos),
        "bytes": sum(i["bytes"] for i in part_infos),
    }
    return feature_stats, part_seconds, took, part_infos


def finish_seconds(
    seconds: dict[str, float],
    sources: dict[str, float],
    part_seconds: dict[str, float],
    wall: float,
) -> dict[str, float]:
    pipeline = sum(seconds.values())
    seconds.update({f"candidates/{k}": v for k, v in sources.items()})
    seconds.update(part_seconds)
    seconds["pipeline"] = pipeline
    seconds["wall"] = time.perf_counter() - wall
    return seconds


def size_result(
    n: int,
    info: Mapping,
    hist,
    cands: pl.DataFrame,
    feature_stats: Mapping,
    dog: Watchdog,
    breakdown: bool,
    seconds: dict[str, float],
) -> dict:
    return {
        "n_users": n,
        "source_path": info["path"],
        "events_loaded": info["loaded_rows"],
        "loaded_users": info["loaded_users"],
        "history_rows": hist.height,
        "history_users": hist["uid"].n_unique(),
        "candidate_rows": cands.height,
        "candidate_rows_per_user": cands.height / max(hist["uid"].n_unique(), 1),
        "candidate_bytes": cands.estimated_size(),
        "feature_rows": feature_stats["rows"],
        "feature_columns": feature_stats["columns"],
        "feature_bytes": feature_stats["bytes"],
        "peak_rss_mb_lifetime": gb.peak_rss_mb(),
        "peak_rss_mb_sampled": dog.peak_bytes / MIB,
        "rss_mb_end": process_rss() / MIB,
        "per_source_separate": breakdown,
        "seconds": seconds,
    }


def add_part_infos(result: dict, part_infos: Sequence[Mapping]) -> None:
    result["feature_parts"] = part_infos
    result["feature_bytes_max_part"] = max((i["bytes"] for i in part_infos), default=0)


def measure_size(
    n: int,
    path: Path,
    uids,
    breakdown: bool,
    state: dict,
    dog: Watchdog,
    parts: Sequence[tuple[str, Sequence]] | None = None,
) -> dict:
    cfg = DEFAULT_CONFIG
    specs = ft.all_specs()
    w = get_window(WINDOW)
    dog.reset_peak()
    seconds: dict[str, float] = {}
    wall = time.perf_counter()
    set_stage(state, n, "load")
    (events, info), seconds["load"] = timed(
        ds.load_protocol_events, "inner", path, uids
    )
    set_stage(state, n, "history")
    hist, seconds["history"] = timed(history, events, w.T)
    sources = candidate_sources(hist, w.T, cfg, n, state, breakdown)
    set_stage(state, n, "candidates")
    (cands, _), seconds["candidates"] = timed(
        ds.get_or_build_candidates, hist, w.T, cfg, False
    )
    if parts is None:
        stats = hand_feature_stats(hist, w.T, cands, cfg, specs, n, state)
    else:
        stats = part_feature_stats(hist, w.T, cands, cfg, parts, n, state, dog)
    feature_stats, part_seconds, seconds["features"], part_infos = stats
    finish_seconds(seconds, sources, part_seconds, wall)
    result = size_result(n, info, hist, cands, feature_stats, dog, breakdown, seconds)
    if parts is not None:
        add_part_infos(result, part_infos)
    del events, hist, cands
    gc.collect()
    return result


def machine_info() -> dict:
    mem = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": mem.total / GIB,
        "memory_available_gb": mem.available / GIB,
        "load_average_start": list(os.getloadavg()),
    }


def plan_scale_size(
    n: int, scale_path: Path, scale_pool, info: dict
) -> tuple[tuple[Path, object], object]:
    if not scale_path.exists():
        info["scale_sample"] = build_scale_sample(SOURCE_PATH, scale_path)
    if scale_pool is None:
        scale_pool = pool_uids(scale_path)
    keep = None if n >= scale_pool.size else nested_sample(scale_pool, n)
    return (scale_path, keep), scale_pool


def scale_sample_info(scale_path: Path) -> dict:
    meta = pl.scan_parquet(scale_path).select(pl.len(), pl.col("uid").n_unique())
    row = meta.collect()
    return {
        "path": str(scale_path),
        "rows": int(row.item(0, 0)),
        "users": int(row.item(0, 1)),
        "bytes": scale_path.stat().st_size,
        "built_by_this_run": False,
    }


def resolve_sources(
    sizes: Sequence[int], scale_path: Path, dev_path: Path
) -> tuple[dict[int, tuple[Path, object]], dict]:
    plan: dict[int, tuple[Path, object]] = {}
    dev_pool = pool_uids(dev_path)
    scale_pool = None
    info: dict = {}
    for n in sizes:
        if n <= dev_pool.size:
            plan[n] = (dev_path, nested_sample(dev_pool, n))
            continue
        plan[n], scale_pool = plan_scale_size(n, scale_path, scale_pool, info)
    if scale_path.exists():
        info.setdefault("scale_sample", scale_sample_info(scale_path))
    return plan, info


def specs_record(
    value: str, parts: Sequence[tuple[str, Sequence]] | None, prereg_path, arm
) -> dict:
    out: dict = {"mode": value}
    if parts is not None:
        out["parts"] = {part: len(specs) for part, specs in parts}
    if prereg_path is not None:
        out["prereg_file"] = Path(prereg_path).name
        out["arm"] = arm
        out["arm_sha256"] = prereg_arm(prereg_path, arm)["sha256"]
    return out


def print_summary(results: Sequence[Mapping], fits: Mapping) -> None:
    for r in results:
        s = r["seconds"]
        print(
            f"n_users {r['n_users']}: history_rows {r['history_rows']} "
            f"candidate_rows {r['candidate_rows']} feature_bytes {r['feature_bytes']} "
            f"pipeline {s['pipeline']:.1f}s wall {s['wall']:.1f}s "
            f"peak_rss_mb {r['peak_rss_mb_lifetime']:.0f}"
        )
        print(
            "  seconds: "
            + ", ".join(
                f"{k} {v:.2f}" for k, v in s.items() if k not in ("pipeline", "wall")
            )
        )
        for info in r.get("feature_parts", ()):
            print(
                f"  part {info['part']}: specs {info['specs']} {info['seconds']:.2f}s "
                f"bytes {info['bytes']} peak_rss_mb {info['peak_rss_mb_sampled']:.0f} "
                f"after_discard_mb {info['rss_mb_after_discard']:.0f}"
            )
    for name in ("seconds/pipeline", "candidate_rows", "peak_rss_mb_lifetime"):
        fit = fits.get(name)
        if fit:
            print(
                f"slope {name}: {fit['slope']:.3f}, "
                f"extrapolated_{SCALE_USERS} {fit[f'extrapolated_{SCALE_USERS}']:.1f}, "
                f"linear {fit[f'extrapolated_linear_{SCALE_USERS}']}, "
                f"measured {fit[f'measured_{SCALE_USERS}']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--max-rss-gb", type=float, default=MAX_RSS_GB)
    parser.add_argument("--skip-10k", action="store_true")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--specs", type=specs_arg, default=SPECS_HAND)
    parser.add_argument("--prereg", default=None)
    return parser


def invalid_sizes(sizes: Sequence[int], max_rss_gb: float) -> bool:
    return not sizes or sizes[0] <= 0 or max_rss_gb <= 0


def validated_sizes(parser: argparse.ArgumentParser, args) -> list[int]:
    sizes = sorted(set(args.sizes))
    if args.skip_10k:
        sizes = [n for n in sizes if n != SCALE_USERS]
    if invalid_sizes(sizes, args.max_rss_gb):
        parser.error("sizes must be positive and non-empty, max-rss-gb must be > 0")
    return sizes


def prereg_path_for(mode: str, prereg_arg: str | None) -> Path | None:
    if mode != SPECS_PREREG:
        return None
    return Path(prereg_arg) if prereg_arg else find_prereg("runs")


def measure_all(
    sizes: Sequence[int],
    plan: Mapping,
    mode: str,
    state: dict,
    dog: Watchdog,
    parts,
) -> None:
    for n in sizes:
        path, uids = plan[n]
        state["completed"].append(
            measure_size(
                n,
                path,
                uids,
                breakdown=mode == SPECS_HAND and n <= DEV_USERS,
                state=state,
                dog=dog,
                parts=parts,
            )
        )


def n_feature_specs(parts) -> int:
    if parts is None:
        return len(ft.all_specs())
    return sum(len(specs_) for _, specs_ in parts)


def scale_notes(parts) -> list[str]:
    return [
        "peak_rss_mb_lifetime is the process maximum so far and is only per-size when sizes ascend",
        "candidates/<source> seconds come from separate single-source build_candidates calls and include shared preprocessing; only for n_users <= 1000",
        "pipeline = load + history + candidates + features; wall also contains the per-source breakdown",
        *(
            []
            if parts is None
            else [
                "features are built part by part (one build_features call per part) and each frame is discarded right after timing; features/prep is the sum of the per-part Context preparation, features is the sum of the part wall times",
            ]
        ),
    ]


def scale_record(
    run_id: str,
    tag: str,
    now: datetime,
    sizes: Sequence[int],
    args,
    parts,
    prereg_path,
    arm,
    data_info: Mapping,
    machine: Mapping,
    results: Sequence[Mapping],
    fits: Mapping,
    start: float,
) -> dict:
    return {
        "run_id": f"{run_id}-scale{tag}",
        "created_at": now.isoformat(timespec="seconds"),
        "seed": SEED,
        "git_commit": gb.GIT_COMMIT,
        "window": {"name": WINDOW, "T": get_window(WINDOW).T},
        "protocol": "inner",
        "scope": "candidates and build_features for one window, events with timestamp < VAL_TS, no labels, use_cache=False",
        "sizes": sizes,
        "fit_sizes": list(FIT_SIZES),
        "max_rss_gb": args.max_rss_gb,
        "watchdog_interval_seconds": WATCH_INTERVAL,
        "sampling": {
            "nested": "sorted uid pool (timestamp < VAL_TS) permuted by numpy default_rng(42), first N, sorted",
            "dev_pool": str(DATA_PATH),
            "scale_pool": str(SCALE_PATH),
            "scale_users_requested": SCALE_USERS,
        },
        "data": data_info,
        "config": {"candidates": DEFAULT_CONFIG.config_dict()},
        "feature_version": ft.FEATURE_VERSION,
        "n_feature_specs": n_feature_specs(parts),
        "specs": specs_record(args.specs, parts, prereg_path, arm),
        "machine": machine,
        "versions": gb.library_versions(),
        "results": results,
        "fits": fits,
        "total_seconds": time.perf_counter() - start,
        "peak_rss_mb_lifetime": gb.peak_rss_mb(),
        "notes": scale_notes(parts),
    }


def write_scale_log(out_dir: Path, run_id: str, tag: str, record: Mapping) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}-scale{tag}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return path


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sizes = validated_sizes(parser, args)
    mode, arm = split_specs(args.specs)
    prereg_path = prereg_path_for(mode, args.prereg)
    parts = feature_parts(mode, arm, prereg_path)
    tag = "" if mode == SPECS_HAND else f"-{args.specs.replace(':', '-')}"
    now = datetime.now()
    run_id = f"{now:%Y%m%d-%H%M%S}"
    out_dir = Path(args.out_dir)
    machine = machine_info()
    print(f"load average at start: {machine['load_average_start']}", flush=True)
    start = time.perf_counter()
    state: dict = {"stage": "sampling", "n_users": None, "completed": []}
    dog = Watchdog(
        args.max_rss_gb * GIB, state, out_dir / f"{run_id}-scale{tag}-partial.json"
    )
    dog.start()
    plan, data_info = resolve_sources(sizes, SCALE_PATH, DATA_PATH)
    results = state["completed"]
    measure_all(sizes, plan, mode, state, dog, parts)
    dog.stop()
    fits = scaling_fits(results)
    print_summary(results, fits)
    record = scale_record(
        run_id,
        tag,
        now,
        sizes,
        args,
        parts,
        prereg_path,
        arm,
        data_info,
        machine,
        results,
        fits,
        start,
    )
    path = write_scale_log(out_dir, run_id, tag, record)
    print(f"run log: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
