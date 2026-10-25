import argparse
import functools
import gc
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import psutil

from recsys_tbank import autofeat as af
from recsys_tbank import baselines as bl
from recsys_tbank import dataset as ds
from recsys_tbank import gbdt as gb
from recsys_tbank import split as sp
from recsys_tbank.candidates import DEFAULT_CONFIG, LABELS, SLICES
from recsys_tbank.split import DATA_PATH, TEST_TS

_RA_PATH = Path(__file__).resolve().with_name("05_run_autofeat.py")
_spec = importlib.util.spec_from_file_location("run_autofeat_final", _RA_PATH)
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "final-test-1"
RUN_NAME = "final-test"
LOG_GLOB = f"*-{RUN_NAME}.json"
GUARD_NAME = "final-test.guard"
GUARD_EXIT_CODE = 4
FROZEN_EXIT_CODE = 5
GATE_EXIT_CODE = 6
GIB = 1024**3
MIN_FREE_RAM_GIB = 7.0
MIN_FREE_DISK_GIB = 10.0
TEST_WINDOW = "test"
VAL_WINDOW = rf.VAL_WINDOW
REFERENCE = rf.VAL_REFERENCE
ARMS = (REFERENCE, *af.PREREG_ARMS)
BASELINE_REF = "B3"
TRAIN_WINDOWS = af.confirm_window_names()
FINAL_TRAIN = (*TRAIN_WINDOWS, VAL_WINDOW)
BOOT_SLICES = rf.BOOT_SLICES
DIGITS = bl.DIGITS
TOL = 10.0**-DIGITS
PER_USER_TOL = rf.VAL_PER_USER_TOL
DATA_KEYS = ("file_rows", "file_bytes", "loaded_rows", "loaded_users", "loaded_filter")
FROZEN = {
    "params": (
        "runs/gbdt-best-params.json",
        "8c1109d44ab0b306eaa4b65700cf2184191b69da5eeab04cd095097e1f7a7c32",
    ),
    "prereg": (
        "runs/20261019-183930-autofeat-prereg.json",
        "1800dc4e7dabf714d4404a4d7e24c0884d22d3e9e5b1fe01c1f53e8183a56a8d",
    ),
    "val_log": (
        "runs/20261019-205929-autofeat-val.json",
        "4062438acbdd1a4b0713ed31256430681d4c445992175bcafa807558a748109b",
    ),
    "e1_log": (
        "runs/20261019-133109-exp-E1.json",
        "d27803854d6d75a514ec9aa3a9aeaf89e512b0c125a570308144a8410bb78583",
    ),
    "baselines_log": (
        "runs/20261018-210235-baselines.json",
        "8b356bcb38cb4c5473917c8c770ebc197bbb69131e72e3b10c5e1c0025818c7a",
    ),
}
CODE_FILES = (
    "scripts/07_final_test.py",
    "scripts/05_run_autofeat.py",
    "src/recsys_tbank/autofeat.py",
    "src/recsys_tbank/baselines.py",
    "src/recsys_tbank/candidates.py",
    "src/recsys_tbank/dataset.py",
    "src/recsys_tbank/gbdt.py",
    "src/recsys_tbank/metrics.py",
    "src/recsys_tbank/split.py",
)
RULES = {
    "gate": f"--execute refuses to start when runs/{GUARD_NAME} or any runs/{LOG_GLOB} exists; "
    "it then checks the sha256 of the frozen files, the pre-registration (autofeat.verify_prereg, "
    "the only accepted difference is the existing val log, plus the val-stage binding checks), "
    "the val log digest and its consistency with the pre-registration, and the frozen baseline "
    "params; any difference stops the run before data is loaded",
    "label_free_gates": "before the guard: events with timestamp < day 294 only (protocol outer); "
    "the val window is rebuilt as in the val stage; the test candidates and feature parts are "
    "prepared from history before day 294 without labels; B0-B6 are replayed on val with the "
    "Phase 2 calibration (train windows) and must equal the E1 log to 6 decimals; every inner fit "
    "(train_266 + train_273, early stopping on train_280) of E1a, G127, G53 and U for seeds 42-46 "
    "must equal the val log in best_iteration and in inner ndcg@10 to 6 decimals",
    "guard": "the guard marker is created exclusively after these gates and before any event at "
    "or after day 294 is read; a crash leaves it in place",
    "rounds": "num_boost_round = round(inner best_iteration x 1.5) of the same arm and seed, "
    "read from the val log for every arm (user decision Q2); not rescaled for the larger refit set",
    "refit": "refit on train_266 + train_273 + train_280 + val without early stopping, frozen "
    "params, features in the pre-registered order; the test scores come from this refit",
    "calibration": "LogLoss probabilities from the percentile bin_calibrator fitted on the "
    "train_280 scores of the inner model of the same arm and seed, as in the val stage; "
    "baselines: bl._calibrate fitted on train_266, train_273, train_280 and val",
    "metrics": "metrics.evaluate through baselines._evaluate_keys on the test window: y_org and "
    "y_all, slices all, novel and repeat, all-users variants; ROC-AUC is pooled over rows and "
    "is not a model comparison metric",
    "bootstrap": "paired user bootstrap of per-user ndcg@10 (y_org), seed-averaged per user, users "
    "with >= 1 positive of the slice, autofeat.paired_bootstrap with the pre-registered n_boot "
    "and seed; G127, G53, U vs E1a at the pre-registered levels, E1a vs B3 at 95%; novel is "
    "descriptive",
    "reading": "the pre-registered H1 rule (lower 95% bound of G127 - E1a > -0.005) is applied "
    "unchanged to the test numbers; G53, U and all other metrics are exploratory",
}


class CheckError(Exception):
    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class GuardError(CheckError):
    pass


class FrozenConfigError(CheckError):
    pass


class GateError(CheckError):
    pass


@dataclass(frozen=True)
class Env:
    root: Path = ROOT
    runs_dir: Path = ROOT / "runs"
    data: Path = DATA_PATH
    cache_dir: Path = Path(af.CACHE_DIR)
    frozen: Mapping[str, tuple[str, str]] = field(default_factory=lambda: dict(FROZEN))


def guard_hits(runs_dir: str | Path) -> list[str]:
    d = Path(runs_dir)
    marker = d / GUARD_NAME
    found = [str(marker)] if marker.exists() else []
    return found + [str(p) for p in sorted(d.glob(LOG_GLOB))]


def claim_guard(runs_dir: str | Path, info: Mapping) -> Path:
    path = Path(runs_dir) / GUARD_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise GuardError([f"{path} already exists"]) from exc
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(bl._clean(info), indent=2) + "\n")
    return path


def feature_registry(params: Mapping) -> tuple[list[str], list[str]]:
    hand = list(gb.resolve_features(gb.GBDTConfig(params=dict(params)), af.hand_specs()))
    known = [s.name for s in (*af.hand_specs(), *af.generated_specs())]
    return hand, known


def code_hashes() -> dict[str, str]:
    return {f: af.file_sha256(ROOT / f) for f in CODE_FILES}


def check_sources(env: Env) -> dict[str, dict]:
    problems: list[str] = []
    if set(env.frozen) != set(FROZEN):
        raise FrozenConfigError([f"frozen roles {sorted(env.frozen)} differ from {sorted(FROZEN)}"])
    out = {}
    for role, (rel, sha) in env.frozen.items():
        path = Path(env.root) / rel
        if not path.is_file():
            problems.append(f"{role}: {rel} is missing")
            continue
        got = af.file_sha256(path)
        if got != sha:
            problems.append(f"{role}: {rel} sha256 {got} differs from the frozen {sha}")
        out[role] = {"path": rel, "sha256": got}
    if problems:
        raise FrozenConfigError(problems)
    return out


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def frozen_arm(
    arm: str, features: Sequence[str], logged: Mapping, seeds: Sequence[int], problems: list[str]
) -> dict:
    feats = list(features)
    if logged.get("n_features") != len(feats) or logged.get("features_sha256") != af.list_sha256(feats):
        problems.append(f"{arm}: val log features differ from the pre-registered list")
    out = {
        "features": feats,
        "n_features": len(feats),
        "features_sha256": af.list_sha256(feats),
        "features_sha256_ordered": af.list_sha256(feats, ordered=True),
        "inner_best_iteration": {},
        f"inner_{gb.STOP_METRIC}": {},
        "val_num_boost_round": {},
        "num_boost_round": {},
    }
    fits = logged.get("fits", {})
    for s in seeds:
        fit = fits.get(f"s{s}", {})
        it = fit.get("inner_best_iteration")
        score = fit.get(f"inner_best_{gb.STOP_METRIC}")
        if fit.get("seed") != s or not _is_int(it) or it <= 0 or not isinstance(score, (int, float)):
            problems.append(f"{arm} s{s}: val log has no valid inner best iteration and score")
            continue
        out["inner_best_iteration"][f"s{s}"] = it
        out[f"inner_{gb.STOP_METRIC}"][f"s{s}"] = float(score)
        out["val_num_boost_round"][f"s{s}"] = fit.get("num_boost_round")
        out["num_boost_round"][f"s{s}"] = round(it * gb.REFIT_FACTOR)
    return out


def cross_check_arms(arms: Mapping[str, dict], binding: Mapping, seeds: Sequence[int]) -> list[str]:
    problems = []
    ref = binding.get("reference", {}).get("per_seed", {})
    e1 = arms[REFERENCE]
    for s in seeds:
        k = f"s{s}"
        if k not in e1["inner_best_iteration"]:
            continue
        want = ref.get(k, {})
        if e1["inner_best_iteration"][k] != want.get("best_iteration"):
            problems.append(f"{REFERENCE} {k}: val log inner best_iteration differs from the pre-registration")
        if not (e1["val_num_boost_round"][k] == e1["num_boost_round"][k] == want.get("num_boost_round")):
            problems.append(f"{REFERENCE} {k}: val rounds do not follow the per-seed rule")
    for a in af.PREREG_ARMS:
        spec = binding.get("arms", {}).get(a, {})
        expected = {f"s{gb.SEED}": spec.get("best_iteration_s42"), **spec.get("best_iteration_other_seeds", {})}
        for k, want in expected.items():
            if arms[a]["inner_best_iteration"].get(k) != want:
                problems.append(f"{a} {k}: val log inner best_iteration differs from the pre-registration")
        for k, used in arms[a]["val_num_boost_round"].items():
            if used != spec.get("num_boost_round"):
                problems.append(f"{a} {k}: val rounds differ from the pre-registered num_boost_round")
    return problems


def preflight(env: Env) -> dict:
    sources = check_sources(env)
    root = Path(env.root)
    path = {role: root / v["path"] for role, v in sources.items()}
    params, raw_source = rf.load_params(path["params"])
    params_source = {"path": sources["params"]["path"], "sha256": raw_source["sha256"]}
    hand, known = feature_registry(params)
    body = json.loads(path["prereg"].read_text())
    val = json.loads(path["val_log"].read_text())
    e1 = json.loads(path["e1_log"].read_text())
    base_log = json.loads(path["baselines_log"].read_text())
    problems: list[str] = []
    verify = af.verify_prereg(path["prereg"], root)
    accepted = [f"{path['val_log'].name} already exists"]
    if verify["diff"] != accepted:
        problems.append(f"prereg verification gave {verify['diff']}, only {accepted} is accepted")
    binding = body.get("binding", {})
    problems += rf.check_binding(binding, hand, known, params_source["sha256"], root)
    if binding.get("params", {}).get("file") != sources["params"]["path"]:
        problems.append("the pre-registered params file is not the frozen params file")
    ref = binding.get("reference", {})
    if (ref.get("source_log"), ref.get("source_log_sha256")) != (
        sources["e1_log"]["path"],
        sources["e1_log"]["sha256"],
    ):
        problems.append("the E1a reference log of the pre-registration is not the frozen e1_log")
    for role, rec, digest in (
        ("val_log", val, gb.metrics_digest),
        ("e1_log", e1, gb.metrics_digest),
        ("baselines_log", base_log, bl.metrics_digest),
    ):
        if rec.get("metrics_digest") != digest(rec):
            problems.append(f"{role}: metrics_digest does not match the file body")
    if val.get("status") != "completed" or not val.get("r0_check", {}).get("all_match"):
        problems.append("val log is not a completed run with a reproduced E1a")
    if not val.get("inner_gate", {}).get("passed"):
        problems.append("val log inner gate did not pass")
    val_parts = val.get("inner_gate", {}).get("prepared_parts", {}).get("parts")
    if not isinstance(val_parts, Mapping) or not val_parts or not all(
        isinstance(v, Mapping) and isinstance(v.get("file"), str) for v in val_parts.values()
    ):
        problems.append("val log has no prepared val part files")
        val_parts = {}
    ref_key = f"{REFERENCE}:s{gb.SEED}"
    val_ref = {
        lb: {sl: val.get("metrics", {}).get(VAL_WINDOW, {}).get(lb, {}).get(sl, {}).get(ref_key) for sl in SLICES}
        for lb in LABELS
    }
    if not all(isinstance(m, Mapping) for by_slice in val_ref.values() for m in by_slice.values()):
        problems.append(f"val log has no val metrics for {ref_key}")
    vp = val.get("config", {}).get("prereg", {})
    if (vp.get("file"), vp.get("file_sha256"), vp.get("decision_digest")) != (
        path["prereg"].name,
        sources["prereg"]["sha256"],
        body.get("decision_digest"),
    ):
        problems.append("val log was not produced from the frozen pre-registration")
    vcfg = val.get("config", {})
    if vcfg.get("params_source", {}).get("sha256") != params_source["sha256"] or vcfg.get("params") != params:
        problems.append("val log params differ from the frozen params file")
    seeds = list(binding.get("val_seeds", []))
    if seeds != list(af.PREREG_SEEDS) or list(val.get("seeds", [])) != seeds:
        problems.append(f"seeds differ: binding {seeds}, val log {val.get('seeds')}, code {list(af.PREREG_SEEDS)}")
    if base_log.get("config", {}).get("baseline_params") != gb.BASELINE_PARAMS.config_dict():
        problems.append("gbdt.BASELINE_PARAMS differ from the frozen baselines log")
    if not gb.BASELINE_PARAMS_SOURCE.startswith(sources["baselines_log"]["path"]):
        problems.append("gbdt.BASELINE_PARAMS_SOURCE does not name the frozen baselines log")
    e1_val = e1.get("metrics", {}).get("outer", {}).get(VAL_WINDOW)
    if not isinstance(e1_val, Mapping):
        problems.append("e1_log has no outer val metrics for the baseline replay")
    if problems:
        raise FrozenConfigError(problems)
    features = {REFERENCE: list(hand)}
    features.update({a: list(binding["arms"][a]["features"]) for a in af.PREREG_ARMS})
    arms = {a: frozen_arm(a, features[a], val.get("arms", {}).get(a, {}), seeds, problems) for a in ARMS}
    for a in af.PREREG_ARMS:
        arms[a]["role"] = binding["arms"][a]["role"]
    arms[REFERENCE]["role"] = "reference"
    if not problems:
        problems += cross_check_arms(arms, binding, seeds)
    if problems:
        raise FrozenConfigError(problems)
    union = list(dict.fromkeys([*hand, *(c for a in af.PREREG_ARMS for c in features[a])]))
    return {
        "sources": sources,
        "params": params,
        "params_source": params_source,
        "prereg_path": path["prereg"],
        "prereg": {
            "file": path["prereg"].name,
            "file_sha256": sources["prereg"]["sha256"],
            "decision_digest": body["decision_digest"],
            "metrics_digest": body["metrics_digest"],
            "verify": verify,
            "accepted_diff": accepted,
        },
        "binding": binding,
        "seeds": seeds,
        "arms": arms,
        "union": union,
        "data_ref": dict(val.get("data", {})),
        "val_log": {"file": path["val_log"].name, "run_id": val.get("run_id"), "metrics_digest": val["metrics_digest"]},
        "e1_val": e1_val,
        "val_parts": rf.parts_files(val_parts),
        "val_ref_metrics": val_ref,
        "baseline_params": gb.BASELINE_PARAMS.config_dict(),
    }


def check_data(info: Mapping, ref: Mapping) -> list[str]:
    return [
        f"data {k}: {info.get(k)!r} differs from the val log {ref.get(k)!r}"
        for k in DATA_KEYS
        if info.get(k) != ref.get(k)
    ]


def load_train_windows(columns: Sequence[str], cache_dir: str | Path) -> dict[str, ds.WindowData]:
    return {w: af.confirm_window_data(w, list(columns), 0, cache_dir) for w in TRAIN_WINDOWS}


def build_val_window(
    events: pl.DataFrame, columns: Sequence[str], cache_dir: str | Path
) -> tuple[ds.WindowData, dict]:
    return rf.build_val_data(events, list(columns), Path(cache_dir))


def build_test_window(
    events: pl.DataFrame,
    columns: Sequence[str],
    cache_dir: str | Path,
    labels: bool = True,
    allow_test: bool = False,
    cfg=DEFAULT_CONFIG,
    hand=None,
    generated=None,
):
    w = sp.get_window(TEST_WINDOW)
    if not labels:
        ds.assert_events_before(events, w.T)
    hand = af.hand_specs() if hand is None else tuple(hand)
    if generated is None:
        generated = {g: af.generated_specs(g) for g in af.GENERATED_GROUPS}
    hist = sp.history(events, w.T)
    sp.assert_no_future(hist, w.T)
    cands, cand_hit = ds.get_or_build_candidates(hist, w.T, cfg, True, cache_dir)
    columns = list(columns)
    wanted = set(columns)
    hand_names = [s.name for s in hand]
    plan = [(af.HAND, hand, [c for c in columns if c in set(hand_names)])]
    for g, specs in generated.items():
        plan.append((g, tuple(specs), [s.name for s in specs if s.name in wanted]))
    if {c for _, _, cols in plan for c in cols} != wanted:
        raise KeyError("some requested columns belong to no part")
    parts, files = {}, {}
    for part, specs, cols in plan:
        if not cols:
            continue
        info = af.build_part(hist, w.T, cands, specs, cfg, True, cache_dir)
        parts[part] = {k: info[k] for k in ("file", "specs", "hit", "seconds")}
        files[part] = (Path(cache_dir) / info["file"], cols)
    n_hist_users = hist["uid"].n_unique()
    if not labels:
        return {"candidates_hit": bool(cand_hit), "parts": parts, "n_hist_users": n_hist_users}
    built = ds.build_window_frame(events, w, cfg, hand, True, cache_dir, allow_test)
    if built.n_hist_users != n_hist_users:
        raise ValueError("history user count mismatch")
    base = built.frame
    keys = base.select(ds.PAIR)
    feats = []
    for part, (path, cols) in files.items():
        if part == af.HAND:
            feats.append(base.select(cols))
            continue
        table = pl.scan_parquet(path).select(*ds.PAIR, *cols).collect()
        if not ds._same_keys(table, keys):
            raise ValueError(f"{part} rows of test do not match the candidate rows")
        feats.append(table.drop(ds.PAIR))
    merged = pl.concat(feats, how="horizontal")
    fixed = [c for c in base.columns if c not in set(hand_names) and c not in af.LABEL_COLUMNS]
    frame = pl.concat(
        [
            base.select(fixed),
            merged.select(pl.col(c).cast(pl.Float32) for c in columns),
            base.select(af.LABEL_COLUMNS),
        ],
        how="horizontal",
    )
    data = built._replace(frame=frame, features=tuple(columns))
    return data, parts


def baseline_frame(events: pl.DataFrame, data: ds.WindowData) -> tuple[pl.DataFrame, pl.DataFrame]:
    return gb.add_baselines(events, data, gb.eval_frame(data))


def _same_value(mine, old) -> bool:
    finite = isinstance(mine, (int, float)) and math.isfinite(mine)
    if old is None or not finite:
        return old is None and not finite
    return abs(float(mine) - float(old)) <= TOL


def evaluate_baselines(frames, lists, datas, names: Sequence[str], target: str, calib: Sequence[str]):
    return gb.evaluate_models(
        {n: frames[n] for n in names},
        {n: datas[n].n_pos for n in names},
        {n: lists[n] for n in names},
        {n: datas[n].n_hist_users for n in names},
        [],
        target,
        calib,
    )


def baseline_replay(frames, lists, datas, logged: Mapping) -> dict:
    metrics, _ = evaluate_baselines(frames, lists, datas, FINAL_TRAIN, VAL_WINDOW, TRAIN_WINDOWS)
    per_baseline = {}
    for b in bl.REPORTED_BASELINES:
        diffs, ok = [], True
        for lb in LABELS:
            for sl in SLICES:
                mine = metrics[VAL_WINDOW][lb][sl][b]
                old = logged.get(lb, {}).get(sl, {}).get(b)
                if not isinstance(old, Mapping) or set(old) != set(mine):
                    ok = False
                    continue
                for m, v in mine.items():
                    ok = ok and _same_value(v, old[m])
                    if isinstance(v, (int, float)) and isinstance(old[m], (int, float)) and math.isfinite(v):
                        diffs.append(abs(float(v) - float(old[m])))
        per_baseline[b] = {"max_abs_diff": max(diffs) if diffs else None, "match": ok}
    return {
        "target": VAL_WINDOW,
        "calibration_windows": list(TRAIN_WINDOWS),
        "tolerance": TOL,
        "per_baseline": per_baseline,
        "all_match": all(v["match"] for v in per_baseline.values()),
    }


def inner_fits(
    datas: Mapping[str, ds.WindowData],
    frozen: Mapping,
    arms: Sequence[str] = ARMS,
    seeds: Sequence[int] | None = None,
) -> dict:
    train = {n: datas[n] for n in TRAIN_WINDOWS}
    out: dict[str, dict[int, dict]] = {}
    for arm in arms:
        feats = frozen["arms"][arm]["features"]
        sub = rf.restrict(train, feats, frozen["union"])
        out[arm] = {}
        for s in frozen["seeds"] if seeds is None else seeds:
            out[arm][s] = rf.inner_fit_arm(sub, frozen["params"], feats, s)
            print(
                f"  inner {arm}:s{s} best_iter {out[arm][s]['best_iteration']} "
                f"ndcg@10 {out[arm][s]['best_score']:.6f}",
                flush=True,
            )
        del sub
        gc.collect()
    return out


def inner_gate(inner: Mapping, frozen: Mapping) -> tuple[list[str], dict]:
    problems, report = [], {}
    for arm in ARMS:
        if arm not in inner:
            continue
        spec = frozen["arms"][arm]
        report[arm] = {}
        for s in frozen["seeds"]:
            if s not in inner[arm]:
                continue
            k = f"s{s}"
            got = inner[arm][s]
            row = {
                "best_iteration": got["best_iteration"],
                "val_log_best_iteration": spec["inner_best_iteration"][k],
                f"inner_{gb.STOP_METRIC}": got["best_score"],
                f"val_log_inner_{gb.STOP_METRIC}": spec[f"inner_{gb.STOP_METRIC}"][k],
                "num_boost_round": spec["num_boost_round"][k],
            }
            same_score = got["best_score"] is not None and round(got["best_score"], DIGITS) == round(
                row[f"val_log_inner_{gb.STOP_METRIC}"], DIGITS
            )
            row["match"] = bool(got["best_iteration"] == row["val_log_best_iteration"] and same_score)
            if not row["match"]:
                problems.append(f"{arm} {k}: inner fit differs from the val log {row}")
            report[arm][k] = row
    return problems, report


def refit_replay(datas: Mapping[str, ds.WindowData], frozen: Mapping, inner: Mapping) -> dict:
    s = gb.SEED
    k = f"s{s}"
    key = f"{REFERENCE}:{k}"
    spec = frozen["arms"][REFERENCE]
    sub = rf.restrict({n: datas[n] for n in (*TRAIN_WINDOWS, VAL_WINDOW)}, spec["features"], frozen["union"])
    scores, probs, _ = rf.refit_arm(
        sub, frozen["params"], spec["features"], s, spec["num_boost_round"][k], inner
    )
    metrics, _ = rf.val_arm_metrics(datas[VAL_WINDOW], key, scores, probs)
    diffs, ok = [], True
    for lb in LABELS:
        for sl in SLICES:
            mine, old = metrics[lb][sl], frozen["val_ref_metrics"][lb][sl]
            if set(mine) != set(old):
                ok = False
                continue
            for m, v in mine.items():
                ok = ok and _same_value(v, old[m])
                if isinstance(v, (int, float)) and isinstance(old[m], (int, float)) and math.isfinite(v):
                    diffs.append(abs(float(v) - float(old[m])))
    return {
        "key": key,
        "train_windows": list(TRAIN_WINDOWS),
        "target": VAL_WINDOW,
        "num_boost_round": spec["num_boost_round"][k],
        "tolerance": TOL,
        "ndcg@10": metrics["y_org"]["all"]["ndcg@10"],
        "val_log_ndcg@10": frozen["val_ref_metrics"]["y_org"]["all"]["ndcg@10"],
        "max_abs_diff": max(diffs) if diffs else None,
        "match": ok,
    }


def prepare(env: Env, frozen: Mapping, fit_inner: bool) -> dict:
    runtime: dict[str, float] = {}
    t0 = time.perf_counter()
    events, info = ds.load_protocol_events("outer", env.data)
    runtime["load_outer"] = time.perf_counter() - t0
    latest = events["timestamp"].max() if events.height else None
    info = {**info, "loaded_max_ts": None if latest is None else int(latest)}
    print(f"loaded_filter: {info['loaded_filter']}, rows {info['loaded_rows']}, max timestamp {latest}", flush=True)
    problems = check_data(info, frozen["data_ref"])
    if problems:
        raise FrozenConfigError(problems)
    t0 = time.perf_counter()
    datas = load_train_windows(frozen["union"], env.cache_dir)
    runtime["load_train_windows"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    datas[VAL_WINDOW], val_parts = build_val_window(events, frozen["union"], env.cache_dir)
    runtime["build_val"] = time.perf_counter() - t0
    print(f"  {VAL_WINDOW} rebuilt {datas[VAL_WINDOW].frame.shape} parts {rf.parts_files(val_parts)}", flush=True)
    t0 = time.perf_counter()
    prepared = build_test_window(events, frozen["union"], env.cache_dir, labels=False)
    runtime["prepare_test_parts"] = time.perf_counter() - t0
    print(
        f"  test candidates and feature parts from events < {TEST_TS}: "
        f"candidates_hit {prepared['candidates_hit']} parts {rf.parts_files(prepared['parts'])}",
        flush=True,
    )
    t0 = time.perf_counter()
    frames, lists = {}, {}
    for n in FINAL_TRAIN:
        frames[n], lists[n] = baseline_frame(events, datas[n])
    replay = baseline_replay(frames, lists, datas, frozen["e1_val"])
    runtime["baseline_replay"] = time.perf_counter() - t0
    print(
        f"baseline replay on {VAL_WINDOW} vs {frozen['sources']['e1_log']['path']}: all_match "
        f"{replay['all_match']} max_abs_diff "
        f"{max((v['max_abs_diff'] or 0.0) for v in replay['per_baseline'].values()):.2e}",
        flush=True,
    )
    t0 = time.perf_counter()
    inner = inner_fits(datas, frozen) if fit_inner else inner_fits(datas, frozen, (REFERENCE,), (gb.SEED,))
    runtime["inner_fits"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    replay_fit = refit_replay(datas, frozen, inner[REFERENCE][gb.SEED])
    runtime["refit_replay"] = time.perf_counter() - t0
    print(
        f"refit replay {replay_fit['key']} on {replay_fit['train_windows']} -> {VAL_WINDOW}: "
        f"ndcg@10 {replay_fit['ndcg@10']:.6f} val log {replay_fit['val_log_ndcg@10']:.6f} "
        f"max_abs_diff {replay_fit['max_abs_diff']:.2e} match {replay_fit['match']}",
        flush=True,
    )
    return {
        "events": events,
        "info": info,
        "datas": datas,
        "val_parts": val_parts,
        "val_parts_files": rf.parts_files(val_parts),
        "prepared": prepared,
        "frames": frames,
        "lists": lists,
        "replay": replay,
        "refit_replay": replay_fit,
        "inner": inner,
        "runtime": runtime,
    }


def windows_config() -> dict:
    return {n: {"T": sp.get_window(n).T, "T_end": sp.get_window(n).T_end} for n in (*FINAL_TRAIN, TEST_WINDOW)}


def print_frozen(frozen: Mapping, env: Env) -> None:
    print("frozen sources (sha256 checked):")
    for role, v in frozen["sources"].items():
        print(f"  {role:14} {v['path']} {v['sha256'][:16]} ok")
    print(f"prereg verify diff {frozen['prereg']['verify']['diff']} accepted; binding checks ok")
    print(f"params {frozen['params']} sha256 {frozen['params_source']['sha256'][:16]}")
    print(f"baseline params {frozen['baseline_params']}")
    for n, w in windows_config().items():
        print(f"  window {n:10} [{w['T']}, {w['T_end']})")
    print("arms: rounds = round(1.5 x val-log inner best_iteration) per seed")
    for a, spec in frozen["arms"].items():
        cells = " ".join(
            f"s{s} {spec['inner_best_iteration'][f's{s}']}->{spec['num_boost_round'][f's{s}']}"
            for s in frozen["seeds"]
        )
        print(f"  {a:5} {spec['role']:9} n={spec['n_features']:3d} sha {spec['features_sha256'][:12]} {cells}")
    print(f"union columns {len(frozen['union'])}, seeds {frozen['seeds']}")
    hits = guard_hits(env.runs_dir)
    print(f"guard: {'present ' + str(hits) + ', --execute would refuse' if hits else 'absent'}")


def _existing(path: str | Path) -> Path:
    p = Path(path).absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_resources(env: Env) -> dict:
    disks = [shutil.disk_usage(_existing(p)).free / GIB for p in (env.runs_dir, env.cache_dir)]
    return {"ram_gib": psutil.virtual_memory().available / GIB, "disk_gib": min(disks)}


def check_resources(env: Env, enforce: bool) -> dict:
    res = free_resources(env)
    problems = []
    if res["ram_gib"] < MIN_FREE_RAM_GIB:
        problems.append(f"available RAM {res['ram_gib']:.1f} GiB is below the required {MIN_FREE_RAM_GIB:g} GiB")
    if res["disk_gib"] < MIN_FREE_DISK_GIB:
        problems.append(f"free disk {res['disk_gib']:.1f} GiB is below the required {MIN_FREE_DISK_GIB:g} GiB")
    print(
        f"resources: available RAM {res['ram_gib']:.1f} GiB (min {MIN_FREE_RAM_GIB:g}), "
        f"free disk {res['disk_gib']:.1f} GiB (min {MIN_FREE_DISK_GIB:g})"
        + (", --execute would refuse" if problems and not enforce else ""),
        flush=True,
    )
    if problems and enforce:
        raise GateError(problems)
    return {**res, "min_ram_gib": MIN_FREE_RAM_GIB, "min_disk_gib": MIN_FREE_DISK_GIB}


def label_free_problems(ctx: Mapping, frozen: Mapping) -> tuple[list[str], dict]:
    problems = []
    if not ctx["replay"]["all_match"]:
        problems.append(f"baseline replay differs: {ctx['replay']['per_baseline']}")
    if ctx["val_parts_files"] != frozen["val_parts"]:
        problems.append(
            f"rebuilt val part files {ctx['val_parts_files']} differ from the val log {frozen['val_parts']}"
        )
    if not ctx["refit_replay"]["match"]:
        problems.append(f"refit replay differs from the val log: {ctx['refit_replay']}")
    gate_problems, gate_report = inner_gate(ctx["inner"], frozen)
    return problems + gate_problems, gate_report


def run_dry(env: Env) -> dict:
    frozen = preflight(env)
    print_frozen(frozen, env)
    check_resources(env, False)
    ctx = prepare(env, frozen, fit_inner=False)
    problems, _ = label_free_problems(ctx, frozen)
    if problems:
        raise GateError(problems)
    print("dry run ok: no event at or after the test start was read, nothing was written to runs/")
    return {"frozen": frozen, **ctx}


def record_base(env: Env, frozen: Mapping, now: datetime, guard: Path, loaded: Sequence[str]) -> dict:
    return {
        "seed": gb.SEED,
        "seeds": list(frozen["seeds"]),
        "git_commit": gb.GIT_COMMIT,
        "stage": RUN_NAME,
        "protocol": "final",
        "selection_window": af.confirm_fold().valid,
        "target_split": TEST_WINDOW,
        "loaded_windows": list(loaded),
        "config": {
            "version": VERSION,
            "frozen_sources": frozen["sources"],
            "code_sha256": code_hashes(),
            "code_identity_prereg": af.code_identity(),
            "prereg": frozen["prereg"],
            "val_log": frozen["val_log"],
            "params": frozen["params"],
            "params_source": frozen["params_source"],
            "fit_config": af.confirm_config(frozen["params"], frozen["seeds"][0]).config_dict(),
            "arms": {
                a: dict(spec) for a, spec in frozen["arms"].items()
            },
            "inner_train_windows": list(af.confirm_fold().train),
            "final_train_windows": list(FINAL_TRAIN),
            "refit_factor": gb.REFIT_FACTOR,
            "windows": windows_config(),
            "test_label_days": (sp.get_window(TEST_WINDOW).T_end - TEST_TS) / sp.DAY,
            "baselines": {
                "params": frozen["baseline_params"],
                "source": gb.BASELINE_PARAMS_SOURCE,
                "seed": frozen["baseline_params"]["seed"],
                "calibration_windows": list(FINAL_TRAIN),
            },
            "bootstrap": {
                "n_boot": frozen["binding"]["bootstrap"]["n_boot"],
                "seed": frozen["binding"]["bootstrap"]["seed"],
                "levels": {
                    "G127 vs E1a": frozen["binding"]["hypotheses"]["H1"]["ci_level"],
                    "G53 vs E1a": frozen["binding"]["hypotheses"]["secondary"]["ci_level"],
                    "U vs E1a": frozen["binding"]["hypotheses"]["H2"]["ci_level"],
                    f"{REFERENCE} vs {BASELINE_REF}": af.CI_LEVEL,
                },
            },
            "rules": RULES,
            "guard": {"file": guard.name, "stem": f"{now:%Y%m%d-%H%M%S}"},
            "users": 0,
        },
        "versions": gb.library_versions(),
    }


def best_effort(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (OSError, ValueError, TypeError) as exc:
            print(f"partial artifact failed in {fn.__name__}: {exc!r}", file=sys.stderr, flush=True)

    return wrapper


class Partial:
    def __init__(self, runs_dir: str | Path, stem: str, guard: Path):
        d = Path(runs_dir)
        self.path = d / f"{stem}-{RUN_NAME}-partial.json"
        self.arrays = d / f"{stem}-{RUN_NAME}-scores"
        self.head = {"status": "in_progress", "stage": "start", "stem": stem, "guard": guard.name}
        self.results: dict[str, dict] = {
            "arms": {},
            "metrics": {},
            "all_users": {},
            "metric_keys": {},
            "per_user_check": {},
            "runtime_seconds": {},
        }

    def stage(self, name: str) -> None:
        self.head["stage"] = name

    @best_effort
    def flush(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(bl._clean({**self.head, **self.results}), indent=2) + "\n")
        os.replace(tmp, self.path)

    @best_effort
    def save_scores(self, key: str, scores, probs: Mapping[str, np.ndarray]) -> None:
        self.arrays.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.arrays / f"{key.replace(':', '-')}.npz",
            scores=np.asarray(scores),
            **{f"p_{lb}": np.asarray(probs[lb]) for lb in LABELS},
        )

    @best_effort
    def save_frame(self, name: str, frame: pl.DataFrame) -> None:
        self.arrays.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(self.arrays / f"{name}.parquet")

    def crash(self) -> None:
        self.head["status"] = "crashed"
        self.head["traceback"] = traceback.format_exc()
        self.flush()
        print(f"partial results: {self.path}", file=sys.stderr, flush=True)

    def finish(self) -> None:
        self.path.unlink(missing_ok=True)


def guard_payload(frozen: Mapping, now: datetime, stem: str, resources: Mapping) -> dict:
    return {
        "created_at": now.isoformat(timespec="seconds"),
        "stem": stem,
        "pid": os.getpid(),
        "val_log": frozen["val_log"]["file"],
        "prereg": frozen["prereg"]["file"],
        "rules": RULES,
        "code_sha256": code_hashes(),
        "frozen_sources": frozen["sources"],
        "refit_factor": gb.REFIT_FACTOR,
        "rounds": {
            a: {
                "features_sha256": spec["features_sha256"],
                "inner_best_iteration": spec["inner_best_iteration"],
                "num_boost_round": spec["num_boost_round"],
            }
            for a, spec in frozen["arms"].items()
        },
        "windows": windows_config(),
        "resources": resources,
        "note": "created after the label-free gates and before any event at or after the "
        "test start is read; remove only after a documented decision",
    }


def run_execute(env: Env, now: datetime | None = None) -> Path:
    hits = guard_hits(env.runs_dir)
    if hits:
        raise GuardError([f"refusing to run, found {hits}"])
    start = time.perf_counter()
    now = now or datetime.now(UTC)
    frozen = preflight(env)
    print_frozen(frozen, env)
    resources = check_resources(env, True)
    ctx = prepare(env, frozen, fit_inner=True)
    problems, gate_report = label_free_problems(ctx, frozen)
    if problems:
        raise GateError(problems)
    print(
        "label-free gates passed: baseline replay, val parts, refit replay and all inner fits "
        "equal the frozen logs",
        flush=True,
    )
    stem = f"{now:%Y%m%d-%H%M%S}"
    guard = claim_guard(env.runs_dir, guard_payload(frozen, now, stem, resources))
    print(f"final test guard written: {guard}", flush=True)
    partial = Partial(env.runs_dir, stem, guard)
    try:
        path, record = execute_test(env, frozen, ctx, now, guard, partial, gate_report, start)
    except BaseException:
        partial.crash()
        raise
    print_result(record)
    print(f"final test log: {path}")
    print(f"metrics_digest: {record['metrics_digest']}")
    return path


def execute_test(
    env: Env,
    frozen: Mapping,
    ctx: dict,
    now: datetime,
    guard: Path,
    partial: Partial,
    gate_report: Mapping,
    start: float,
) -> tuple[Path, dict]:
    del ctx["events"]
    gc.collect()
    datas, frames, lists = ctx["datas"], ctx["frames"], ctx["lists"]
    runtime = ctx["runtime"]
    res = partial.results
    arms, metrics, all_users, keys_info, checks = (
        res["arms"], res["metrics"], res["all_users"], res["metric_keys"], res["per_user_check"]
    )
    res["runtime_seconds"] = runtime
    vectors: dict[str, dict] = {}
    uids: dict[str, np.ndarray] = {}
    partial.stage("load_final")
    t0 = time.perf_counter()
    events, final_info = ds.load_protocol_events("final", env.data, allow_test=True)
    runtime["load_final"] = time.perf_counter() - t0
    partial.stage("build_test")
    t0 = time.perf_counter()
    test, test_parts = build_test_window(events, frozen["union"], env.cache_dir, labels=True, allow_test=True)
    datas[TEST_WINDOW] = test
    runtime["build_test"] = time.perf_counter() - t0
    parts_match = rf.parts_files(test_parts) == rf.parts_files(ctx["prepared"]["parts"])
    print(f"  built {TEST_WINDOW} {test.frame.shape} users {test.n_hist_users} parts_match {parts_match}", flush=True)
    label = af.CONFIRM_LABEL

    def add_vectors(name: str, frame: pl.DataFrame, scores) -> dict:
        out = {}
        for sl in BOOT_SLICES:
            u, v = af.per_user_ndcg(frame, scores, test.n_pos[label], label, sl)
            if sl in uids and not np.array_equal(uids[sl], u):
                raise ValueError("per-user vectors are not aligned")
            uids[sl] = u
            out[sl] = v
            checks[f"{sl}/{name}"] = abs(float(v.mean()) - metrics[name][label][sl]["ndcg@10"])
        return out

    partial.stage("test_baselines")
    t0 = time.perf_counter()
    frames[TEST_WINDOW], lists[TEST_WINDOW] = baseline_frame(events, test)
    base_metrics, base_all = evaluate_baselines(
        frames, lists, datas, (*FINAL_TRAIN, TEST_WINDOW), TEST_WINDOW, FINAL_TRAIN
    )
    for b in bl.REPORTED_BASELINES:
        metrics[b] = {lb: {sl: base_metrics[TEST_WINDOW][lb][sl][b] for sl in SLICES} for lb in LABELS}
        all_users[b] = {lb: {sl: base_all[TEST_WINDOW][lb][sl][b] for sl in SLICES} for lb in LABELS}
        keys_info[b] = {"arm": b, "seed": frozen["baseline_params"]["seed"]}
    b3 = add_vectors(BASELINE_REF, frames[TEST_WINDOW], frames[TEST_WINDOW][BASELINE_REF].to_numpy())
    runtime["baselines"] = time.perf_counter() - t0
    partial.save_frame("test-baselines", frames[TEST_WINDOW])
    partial.flush()
    print(f"  test baselines done: B3 ndcg@10 {metrics[BASELINE_REF][label]['all']['ndcg@10']:.6f}", flush=True)
    del events
    gc.collect()
    for arm in ARMS:
        spec = frozen["arms"][arm]
        t0 = time.perf_counter()
        sub = rf.restrict(datas, spec["features"], frozen["union"])
        for s in frozen["seeds"]:
            k = f"s{s}"
            key = f"{arm}:{k}"
            partial.stage(f"refit:{key}")
            scores, probs, info = rf.refit_arm(
                sub, frozen["params"], spec["features"], s, spec["num_boost_round"][k],
                ctx["inner"][arm][s], FINAL_TRAIN, TEST_WINDOW,
            )
            info["val_log_inner_best_iteration"] = spec["inner_best_iteration"][k]
            info["val_num_boost_round"] = spec["val_num_boost_round"][k]
            partial.save_scores(key, scores, probs)
            t1 = time.perf_counter()
            metrics[key], all_users[key] = rf.val_arm_metrics(test, key, scores, probs)
            info["metric_seconds"] = time.perf_counter() - t1
            vectors.setdefault(arm, {})[s] = add_vectors(key, test.frame, scores)
            arms.setdefault(arm, {"fits": {}})["fits"][k] = info
            keys_info[key] = {"arm": arm, "seed": s, "num_boost_round": spec["num_boost_round"][k]}
            partial.flush()
            print(
                f"  {key:10} ndcg@10 {metrics[key]['y_org']['all']['ndcg@10']:.6f} "
                f"inner_best_iter {info['inner_best_iteration']} rounds {info['num_boost_round']}",
                flush=True,
            )
            del scores, probs
            gc.collect()
        del sub
        arms[arm].update(
            {k: spec[k] for k in ("role", "n_features", "features_sha256", "features_sha256_ordered")}
        )
        arms[arm]["seeds"] = list(frozen["seeds"])
        arms[arm]["num_boost_round"] = spec["num_boost_round"]
        runtime[f"fit_{arm}"] = time.perf_counter() - t0
    partial.stage("bootstrap")
    bad = sorted(k for k, v in checks.items() if v > PER_USER_TOL)
    seeds = frozen["seeds"]
    means = {a: {sl: np.mean([vectors[a][s][sl] for s in seeds], axis=0) for sl in BOOT_SLICES} for a in ARMS}
    boot = rf.val_bootstrap(means, frozen["binding"])
    nb = frozen["binding"]["bootstrap"]
    for sl in BOOT_SLICES:
        boot[sl][f"{REFERENCE} vs {BASELINE_REF}"] = af.paired_bootstrap(
            means[REFERENCE][sl], b3[sl], nb["n_boot"], nb["seed"], af.CI_LEVEL
        )
    reading = rf.val_decision(boot["all"], frozen["binding"])
    reading["note"] = RULES["reading"]
    res["bootstrap"] = {"slices": boot, "prereg_rule_on_test": reading}
    partial.flush()
    partial.stage("summaries")
    model_keys = [k for k in metrics if k not in bl.REPORTED_BASELINES]
    order = [*model_keys, *bl.REPORTED_BASELINES]
    stats = {
        a: gb.seed_stats({TEST_WINDOW: rf.grid_by_label(metrics, order)}, [f"{a}:s{s}" for s in seeds])[TEST_WINDOW]
        for a in ARMS
    }
    for a in ARMS:
        arms[a]["summary"] = rf.summarize(metrics, [f"{a}:s{s}" for s in seeds])
    runtime["total"] = time.perf_counter() - start
    partial.stage("write_log")
    record = {
        **record_base(env, frozen, now, guard, [*FINAL_TRAIN, TEST_WINDOW]),
        "status": "completed" if not bad else "per_user_mismatch",
        "data": {"outer": ctx["info"], "final": final_info, "data_sha256": af.file_sha256(env.data)},
        "preflight": {
            "frozen_checks": "passed",
            "baseline_replay": ctx["replay"],
            "refit_replay": ctx["refit_replay"],
            "val_parts": ctx["val_parts_files"],
            "inner_gate": {"passed": True, "report": gate_report},
            "prepared_test_parts": ctx["prepared"],
            "post_guard_parts_match": parts_match,
        },
        "arms": arms,
        "metric_keys": keys_info,
        "n_hist_users": {TEST_WINDOW: test.n_hist_users},
        "n_users": {TEST_WINDOW: {lb: {sl: metrics["B0"][lb][sl]["n_users"] for sl in SLICES} for lb in LABELS}},
        "metrics_layout": "metrics[split][label][slice][arm:seed or baseline] -> metric dict",
        "metrics": {TEST_WINDOW: rf.grid_by_label(metrics, order)},
        "metrics_all_users": {TEST_WINDOW: rf.grid_by_label(all_users, order)},
        "seed_stats": stats,
        "bootstrap": {"rule": RULES["bootstrap"], "label": label, "metric": af.PREREG_METRIC, "slices": boot},
        "prereg_rule_on_test": reading,
        "per_user_check": checks,
        "per_user_mismatch": bad,
        "runtime_seconds": runtime,
        "resources": {"peak_rss_mb_lifetime": gb.peak_rss_mb()},
    }
    path, digest = gb.write_run_log(record, RUN_NAME, env.runs_dir, now)
    partial.finish()
    return path, {**record, "metrics_digest": digest}


def print_result(record: Mapping) -> None:
    test = record["metrics"][TEST_WINDOW]["y_org"]["all"]
    print("test (y_org, slice all): key ndcg@10 recall@10 roc_auc logloss")
    for key in test:
        row = test[key]
        cells = " ".join("n/a" if row.get(m) is None else f"{row[m]:.4f}" for m in ("ndcg@10", "recall@10", "roc_auc", "logloss"))
        print(f"  {key:10} {cells}")
    for sl, pairs in record["bootstrap"]["slices"].items():
        for name, r in pairs.items():
            print(f"  boot {sl:6} {name:12} {r['mean_delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] level {r['ci_level']}")
    h1 = record["prereg_rule_on_test"]["H1"]
    print(f"H1 rule on test: holds {h1['holds']} lower bound {h1['ci_low']:+.4f} vs {h1['threshold']:+.4f}")


def main(argv=None, env: Env | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    env = Env() if env is None else env
    try:
        if args.execute:
            run_execute(env)
        else:
            run_dry(env)
    except GuardError as exc:
        print(f"FINAL TEST GUARD: {exc}", file=sys.stderr)
        return GUARD_EXIT_CODE
    except FrozenConfigError as exc:
        print("FINAL TEST FROZEN CONFIG: mismatch, nothing was run on test", file=sys.stderr)
        for p in exc.problems:
            print(f"  {p}", file=sys.stderr)
        return FROZEN_EXIT_CODE
    except GateError as exc:
        print("FINAL TEST GATE: label-free checks failed, the guard was not claimed", file=sys.stderr)
        for p in exc.problems:
            print(f"  {p}", file=sys.stderr)
        return GATE_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
