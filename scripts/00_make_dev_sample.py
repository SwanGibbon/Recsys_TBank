import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "yambda"
DEFAULT_SRC = RAW_ROOT / "flat" / "50m" / "multi_event.parquet"
DEFAULT_OUT_DIR = ROOT / "data" / "dev"
SEED = 42
N_USERS = 1000


def output_path(out_dir, n_users):
    return Path(out_dir) / f"events_{n_users}u.parquet"


def check_out_dir(out_dir):
    resolved = Path(out_dir).resolve()
    if resolved == RAW_ROOT or RAW_ROOT in resolved.parents:
        raise ValueError(f"{out_dir} is inside {RAW_ROOT}, which is read-only")


def make_sample(src, out_dir, n_users=N_USERS, seed=SEED, force=False):
    src = Path(src)
    check_out_dir(out_dir)
    if not src.is_file():
        raise FileNotFoundError(src)
    out = output_path(out_dir, n_users)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists, pass --force to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(src)
    users = lf.select("uid").unique().sort("uid").collect().sample(n=n_users, seed=seed)
    lf.join(users.lazy(), on="uid", how="semi").sink_parquet(out)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-users", type=int, default=N_USERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        out = make_sample(args.src, args.out_dir, args.n_users, args.seed, args.force)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    dev = pl.read_parquet(out)
    print(out)
    print(dev["uid"].n_unique(), dev.height, dev["uid"].sort().head(5).to_list())
    return 0


if __name__ == "__main__":
    sys.exit(main())
