"""Analyze benchmark CSV results.

Reads benchmark_results.csv and prints aggregated summaries:
- Win-rate matrix by depth pair
- Efficiency comparison (nodes, prunes, time) by depth
- Cross-run comparison when multiple run_ids exist (e.g. different weights)

Usage:
    python analyze.py                          # analyze benchmark_results.csv
    python analyze.py --csv my_results.csv     # custom file
    python analyze.py --run-id 20260415_1337   # filter to one run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict


def load_csv(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        print(f"Error: {path} not found.")
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Win-rate matrix ──────────────────────────────────────────────────────

def print_winrate_matrix(rows: list[dict[str, str]]) -> None:
    """Print a Black-depth × White-depth win-rate table."""
    # Collect (bd, wd) -> {Black: n, White: n, Draw: n}
    data: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = (int(r["black_depth"]), int(r["white_depth"]))
        data[key][r["winner"]] += 1

    if not data:
        print("No data.")
        return

    b_depths = sorted({k[0] for k in data})
    w_depths = sorted({k[1] for k in data})

    print("\n=== Win-rate matrix (Black win%) ===")
    header = "B\\W  " + "".join(f"{'d=' + str(d):>10}" for d in w_depths)
    print(header)
    print("-" * len(header))

    for bd in b_depths:
        parts = [f"d={bd:<3}"]
        for wd in w_depths:
            counts = data.get((bd, wd))
            if counts is None:
                parts.append(f"{'--':>10}")
            else:
                total = sum(counts.values())
                bw = counts.get("Black", 0)
                parts.append(f"{bw/total*100:>8.1f}% ")
        print("".join(parts))

    print()


# ── Efficiency summary ───────────────────────────────────────────────────

def print_efficiency(rows: list[dict[str, str]]) -> None:
    """Print average efficiency stats grouped by (side, depth)."""
    # side -> depth -> lists of values
    Acc = dict[str, list[float]]
    stats: dict[str, dict[int, Acc]] = {
        "Black": defaultdict(lambda: defaultdict(list)),
        "White": defaultdict(lambda: defaultdict(list)),
    }

    for r in rows:
        bd, wd = int(r["black_depth"]), int(r["white_depth"])
        stats["Black"][bd]["avg_nodes"].append(float(r["b_avg_nodes"]))
        stats["Black"][bd]["avg_prunes"].append(float(r["b_avg_prunes"]))
        stats["Black"][bd]["avg_time"].append(float(r["b_avg_time"]))
        stats["White"][wd]["avg_nodes"].append(float(r["w_avg_nodes"]))
        stats["White"][wd]["avg_prunes"].append(float(r["w_avg_prunes"]))
        stats["White"][wd]["avg_time"].append(float(r["w_avg_time"]))

    print("=== Efficiency by depth (averaged across all matchups) ===")
    print(f"{'Side':<8}{'Depth':>6}{'Avg Nodes':>12}{'Avg Prunes':>12}{'Avg Time(s)':>14}")
    print("-" * 52)

    for side in ("Black", "White"):
        for depth in sorted(stats[side]):
            d = stats[side][depth]
            avg_n = sum(d["avg_nodes"]) / len(d["avg_nodes"])
            avg_p = sum(d["avg_prunes"]) / len(d["avg_prunes"])
            avg_t = sum(d["avg_time"]) / len(d["avg_time"])
            print(f"{side:<8}{depth:>6}{avg_n:>12.0f}{avg_p:>12.1f}{avg_t:>14.4f}")

    print()


# ── Cross-run comparison ─────────────────────────────────────────────────

def print_run_comparison(rows: list[dict[str, str]]) -> None:
    """Compare different runs (e.g. different weight configs)."""
    runs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        runs[r["run_id"]].append(r)

    if len(runs) < 2:
        print("(Only one run found — skipping cross-run comparison.)\n")
        return

    print("=== Cross-run comparison ===")
    print(f"{'Run ID':<20}{'Games':>7}{'B Wins':>8}{'W Wins':>8}{'Draws':>8}{'Avg Nodes/move':>16}{'Weights (sample)':>40}")
    print("-" * 107)

    for rid, rrows in sorted(runs.items()):
        bw = sum(1 for r in rrows if r["winner"] == "Black")
        ww = sum(1 for r in rrows if r["winner"] == "White")
        dw = sum(1 for r in rrows if r["winner"] == "Draw")
        all_nodes = [float(r["b_avg_nodes"]) for r in rrows] + [float(r["w_avg_nodes"]) for r in rrows]
        avg_n = sum(all_nodes) / len(all_nodes) if all_nodes else 0
        weights_sample = rrows[0].get("weights", "")[:38]
        print(f"{rid:<20}{len(rrows):>7}{bw:>8}{ww:>8}{dw:>8}{avg_n:>16.0f}  {weights_sample}")

    print()


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Gomoku benchmark CSV")
    p.add_argument("--csv", default="benchmark_results.csv", help="CSV file to analyze")
    p.add_argument("--run-id", default=None, help="Filter to a specific run_id")
    args = p.parse_args()

    rows = load_csv(args.csv)
    if args.run_id:
        rows = [r for r in rows if r["run_id"] == args.run_id]
        if not rows:
            print(f"No data for run_id={args.run_id}")
            sys.exit(1)
        print(f"Filtered to run_id={args.run_id}  ({len(rows)} games)\n")

    print(f"Loaded {len(rows)} game records from {args.csv}\n")

    print_winrate_matrix(rows)
    print_efficiency(rows)
    print_run_comparison(rows)


if __name__ == "__main__":
    main()
