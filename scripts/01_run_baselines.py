import argparse
import math
import sys
import time
from pathlib import Path

from recsys_tbank import baselines as bl

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


SLICES = ("all", "novel")


def fmt(value) -> str:
    if value is None:
        return "n/a"
    return "nan" if math.isnan(value) else f"{value:.4f}"


def print_block(record: dict, key: str, columns, title: str) -> None:
    val = record[key][bl.SELECT_WINDOW]
    n_hist = record["n_hist_users"][bl.SELECT_WINDOW]
    print(f"{bl.SELECT_WINDOW}, {title}, seed {record['seed']}, n_hist_users {n_hist}")
    head = f"{'label':6} {'slice':6} {'base':4} " + " ".join(
        f"{c:>14}" for c in columns
    )
    print(head)
    for label in ("y_org", "y_all"):
        for sl in SLICES:
            for b in bl.REPORTED_BASELINES:
                row = val[label][sl][b]
                cells = " ".join(f"{fmt(row.get(c)):>14}" for c in columns)
                print(f"{label:6} {sl:6} {b:4} {cells}")


def print_table(record: dict) -> None:
    print_block(record, "metrics", COLUMNS, "metric users")
    print_block(record, "metrics_all_users", bl.ALL_USERS_METRICS, "all users")
    print("W table (val, slice all):")
    for row in record["config"]["w_table"]:
        print(
            f"  {row['variant']:3} W={row['W']:>2}  "
            f"y_org ndcg@10 {fmt(row['y_org/ndcg@10'])}  "
            f"y_all ndcg@10 {fmt(row['y_all/ndcg@10'])}"
        )
    print("chosen:", record["config"]["chosen"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baselines")
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    events, info = bl.load_dev_events()
    load_seconds = time.perf_counter() - t0
    result = bl.run_pipeline(events, data=info)
    record = result.record
    record["runtime_seconds"] = {"load": load_seconds, **record["runtime_seconds"]}
    record["runtime_seconds"]["total"] += load_seconds
    path, digest = bl.write_run_log(record, args.name, Path(args.out_dir))
    print_table(record)
    print(f"runtime_seconds total {record['runtime_seconds']['total']:.1f}")
    print(f"run log: {path}")
    print(f"metrics_digest: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
