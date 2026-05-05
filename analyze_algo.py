"""Visualise Minimax vs Alpha-Beta benchmark results.

Usage:
    python analyze_algo.py                         # auto-picks newest benchmark_algo_*.csv
    python analyze_algo.py benchmark_algo_XYZ.csv  # explicit file
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

PLOTS_DIR = "results/plots"
ALGO_DIR  = "results/benchmark_algo"

def _latest_csv() -> str:
    files = sorted(glob.glob(os.path.join(ALGO_DIR, "benchmark_algo_*.csv")))
    if not files:
        # fallback: current directory
        files = sorted(glob.glob("benchmark_algo_*.csv"))
    if not files:
        raise FileNotFoundError("No benchmark_algo_*.csv file found.")
    return files[-1]


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_by_depth(rows: list[dict]) -> dict[int, dict]:
    """Average all numeric metrics per depth level."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[int(r["depth"])].append(r)

    out: dict[int, dict] = {}
    for depth, group in sorted(buckets.items()):
        def avg(key: str) -> float:
            return sum(float(r[key]) for r in group) / len(group)

        out[depth] = {
            "n_games":          len(group),
            "mm_avg_nodes":     avg("mm_avg_nodes"),
            "ab_avg_nodes":     avg("ab_avg_nodes"),
            "mm_avg_time":      avg("mm_avg_time"),
            "ab_avg_time":      avg("ab_avg_time"),
            "ab_pruning_rate":  avg("ab_pruning_rate"),   # fraction 0-1
            "node_red_pct":     avg("node_reduction_pct"),
            "time_red_pct":     avg("time_reduction_pct"),
            "mm_avg_ebf":       avg("mm_avg_ebf"),
            "ab_avg_ebf":       avg("ab_avg_ebf"),
            "speedup":          avg("mm_avg_time") / avg("ab_avg_time")
                                if avg("ab_avg_time") > 0 else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

PALETTE = {
    "mm": "#E07B39",   # orange  – Minimax
    "ab": "#3A86C8",   # blue    – Alpha-Beta
    "red": "#2CA02C",  # green   – reduction / speedup
}


def plot(data: dict[int, dict], csv_path: str) -> None:
    depths = sorted(data.keys())
    xs = np.arange(len(depths))
    bar_w = 0.35

    mm_nodes  = [data[d]["mm_avg_nodes"]    for d in depths]
    ab_nodes  = [data[d]["ab_avg_nodes"]    for d in depths]
    mm_time   = [data[d]["mm_avg_time"]     for d in depths]
    ab_time   = [data[d]["ab_avg_time"]     for d in depths]
    prune_pct = [data[d]["ab_pruning_rate"] * 100 for d in depths]
    node_red  = [data[d]["node_red_pct"]    for d in depths]
    time_red  = [data[d]["time_red_pct"]    for d in depths]
    speedup   = [data[d]["speedup"]         for d in depths]
    mm_ebf    = [data[d]["mm_avg_ebf"]      for d in depths]
    ab_ebf    = [data[d]["ab_avg_ebf"]      for d in depths]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Minimax vs Alpha-Beta — {os.path.basename(csv_path)}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    depth_labels = [f"d={d}" for d in depths]

    # ── 1. Nodes per move ──────────────────────────────────────────────
    ax = axes[0, 0]
    b1 = ax.bar(xs - bar_w / 2, mm_nodes, bar_w, label="Minimax",     color=PALETTE["mm"], alpha=0.85)
    b2 = ax.bar(xs + bar_w / 2, ab_nodes, bar_w, label="Alpha-Beta",  color=PALETTE["ab"], alpha=0.85)
    ax.set_title("Avg Nodes per Move")
    ax.set_ylabel("nodes")
    ax.set_xticks(xs); ax.set_xticklabels(depth_labels)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _add_bar_labels(ax, b1); _add_bar_labels(ax, b2)

    # ── 2. Avg time per move ───────────────────────────────────────────
    ax = axes[0, 1]
    b1 = ax.bar(xs - bar_w / 2, mm_time, bar_w, label="Minimax",    color=PALETTE["mm"], alpha=0.85)
    b2 = ax.bar(xs + bar_w / 2, ab_time, bar_w, label="Alpha-Beta", color=PALETTE["ab"], alpha=0.85)
    ax.set_title("Avg Time per Move (s)")
    ax.set_ylabel("seconds")
    ax.set_xticks(xs); ax.set_xticklabels(depth_labels)
    ax.legend()
    _add_bar_labels(ax, b1, fmt="{:.3f}"); _add_bar_labels(ax, b2, fmt="{:.3f}")

    # ── 3. Speedup factor ─────────────────────────────────────────────
    ax = axes[0, 2]
    bars = ax.bar(xs, speedup, color=PALETTE["red"], alpha=0.85)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Speedup Factor  (mm_time / ab_time)")
    ax.set_ylabel("× faster")
    ax.set_xticks(xs); ax.set_xticklabels(depth_labels)
    _add_bar_labels(ax, bars, fmt="{:.2f}×")

    # ── 4. Node reduction % ───────────────────────────────────────────
    ax = axes[1, 0]
    bars = ax.bar(xs, node_red, color=PALETTE["ab"], alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_title("Node Reduction  (Alpha-Beta vs Minimax)")
    ax.set_ylabel("% fewer nodes")
    ax.set_xticks(xs); ax.set_xticklabels(depth_labels)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _add_bar_labels(ax, bars, fmt="{:.1f}%")

    # ── 5. Alpha-Beta pruning rate ────────────────────────────────────
    ax = axes[1, 1]
    bars = ax.bar(xs, prune_pct, color=PALETTE["ab"], alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_title("Alpha-Beta Pruning Rate")
    ax.set_ylabel("pruned / (nodes + pruned)  %")
    ax.set_xticks(xs); ax.set_xticklabels(depth_labels)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _add_bar_labels(ax, bars, fmt="{:.1f}%")

    # ── 6. Effective branching factor ─────────────────────────────────
    ax = axes[1, 2]
    ax.plot(depth_labels, mm_ebf, "o-", color=PALETTE["mm"], label="Minimax",    linewidth=2, markersize=7)
    ax.plot(depth_labels, ab_ebf, "s-", color=PALETTE["ab"], label="Alpha-Beta", linewidth=2, markersize=7)
    ax.set_title("Effective Branching Factor (EBF)")
    ax.set_ylabel("EBF")
    ax.legend()
    for xi, (ym, ya) in enumerate(zip(mm_ebf, ab_ebf)):
        ax.annotate(f"{ym:.1f}", (depth_labels[xi], ym), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color=PALETTE["mm"])
        ax.annotate(f"{ya:.1f}", (depth_labels[xi], ya), textcoords="offset points",
                    xytext=(8, -12), fontsize=8, color=PALETTE["ab"])

    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(PLOTS_DIR, stem + "_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.show()


def _add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:,.0f}") -> None:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.01,
                fmt.format(h),
                ha="center", va="bottom", fontsize=7.5,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else _latest_csv()
    print(f"Loading: {csv_path}")
    rows = load(csv_path)
    data = aggregate_by_depth(rows)
    print(f"Depths found: {sorted(data.keys())}  ({sum(v['n_games'] for v in data.values())} games total)")
    plot(data, csv_path)
