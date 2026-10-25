import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from recsys_tbank.features import (
    FEATURE_VERSION,
    FeatureSpec,
    all_specs,
    build_features,
    check_inputs,
)
from recsys_tbank.features.indirect import factor_cache
from recsys_tbank.features.meta import mapping_sha256
from recsys_tbank.features.prep import PAIR
from recsys_tbank.features.source import SOURCE_COLUMNS
from recsys_tbank.split import Frame

CACHE_DIR = Path(__file__).resolve().parents[3] / "cache"
PACKAGE_DIR = Path(__file__).resolve().parent


def source_digest(package_dir: str | Path = PACKAGE_DIR) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(package_dir).glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_sum(frame: pl.DataFrame) -> int:
    return int(frame.hash_rows().to_numpy().sum(dtype=np.uint64))


def hist_fingerprint(hist: Frame) -> dict:
    frame = hist.collect() if isinstance(hist, pl.LazyFrame) else hist
    latest = frame["timestamp"].max() if frame.height else None
    return {
        "height": frame.height,
        "max_ts": None if latest is None else int(latest),
        "hash_sum": _hash_sum(frame),
    }


def cands_fingerprint(cands: pl.DataFrame) -> dict:
    present = [c for c in SOURCE_COLUMNS if c in cands.columns]
    keys = cands.select(
        *(pl.col(c).cast(pl.UInt32) for c in PAIR), *present
    ).with_row_index()
    return {"height": keys.height, "hash_sum": _hash_sum(keys)}


def cache_key(
    hist: Frame,
    T: int,
    cands: pl.DataFrame,
    specs: Sequence[FeatureSpec] | None = None,
    cand_config=None,
) -> str:
    specs = all_specs() if specs is None else tuple(specs)
    if cand_config is not None and hasattr(cand_config, "config_dict"):
        cand_config = cand_config.config_dict()
    payload = {
        "T": int(T),
        "hist": hist_fingerprint(hist),
        "cands": cands_fingerprint(cands),
        "cand_config": cand_config,
        "feature_version": FEATURE_VERSION,
        "source_sha256": source_digest(),
        "mapping_sha256": mapping_sha256(),
        "specs": [asdict(s) for s in specs],
        "polars": pl.__version__,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def cache_path(key: str, cache_dir: str | Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{key}.parquet"


def get_or_build(
    hist: Frame,
    T: int,
    cands: pl.DataFrame,
    specs: Sequence[FeatureSpec] | None = None,
    use_cache: bool = True,
    cache_dir: str | Path = CACHE_DIR,
    cand_config=None,
) -> pl.DataFrame:
    T = check_inputs(hist, T, cands)
    if not use_cache:
        return build_features(hist, T, cands, specs)
    path = cache_path(cache_key(hist, T, cands, specs, cand_config), cache_dir)
    if path.exists():
        return pl.read_parquet(path)
    with factor_cache(cache_dir):
        out = build_features(hist, T, cands, specs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    out.write_parquet(tmp)
    os.replace(tmp, path)
    return out
