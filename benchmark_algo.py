"""Algorithm comparison benchmark: Minimax vs Alpha-Beta at equal depth.

Runs N games per depth level with one AI using plain minimax and the other
using alpha-beta (+ TT + Killer + History).  Both sides search to the same
depth so the only variable is the search algorithm.

The key metrics reported are:
  - nodes explored per move
  - pruning rate (alpha-beta only; minimax is always 0%)
  - effective branching factor (EBF)
  - time per move
  - winner (to confirm alpha-beta quality is not worse)

Usage:
    python benchmark_algo.py                      # depths 1-3, 3 games each
    python benchmark_algo.py --depths 2 3 4 --games 5
    python benchmark_algo.py --depth 3 --games 10   # single depth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from typing import Callable, Optional

from gomoku.board import BLACK, BOARD_SIZE, EMPTY, WHITE
from gomoku.game import Game
from gomoku.heuristic import get_weights, reset_weights
from gomoku.search import (
    alphabeta,
    minimax,
    get_node_count,
    get_prune_count,
    reset_counters,
    reset_game_state,
)

# ---------------------------------------------------------------------------
# Output file names (stamped per run)
# ---------------------------------------------------------------------------
OUT_DIR  = "results/benchmark_algo"
TXT_FILE = OUT_DIR + "/benchmark_algo_{run_id}.txt"
CSV_FILE = OUT_DIR + "/benchmark_algo_{run_id}.csv"

CSV_COLUMNS = [
    "run_id",
    "depth",
    "game_num",
    "mm_role",          # which colour minimax played (Black/White)
    "winner",
    "total_moves",
    "game_time_s",
    # Minimax side
    "mm_moves",
    "mm_total_nodes",
    "mm_avg_nodes",
    "mm_max_nodes",
    "mm_total_time",
    "mm_avg_time",
    "mm_max_time",
    "mm_pruning_rate",  # always 0 – kept for uniform schema
    "mm_avg_ebf",
    # Alpha-Beta side
    "ab_moves",
    "ab_total_nodes",
    "ab_avg_nodes",
    "ab_max_nodes",
    "ab_total_time",
    "ab_avg_time",
    "ab_max_time",
    "ab_total_prunes",
    "ab_pruning_rate",
    "ab_avg_ebf",
    # Speedup ratio
    "node_reduction_pct",   # (mm_nodes - ab_nodes) / mm_nodes * 100
    "time_reduction_pct",
    "weights",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player_label(player: int) -> str:
    return {BLACK: "Black", WHITE: "White", EMPTY: "Draw"}.get(player, "?")


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ebf(total_nodes: float, num_moves: int, depth: int) -> float:
    if num_moves == 0 or depth <= 0 or total_nodes <= 0:
        return 0.0
    return (total_nodes / num_moves) ** (1.0 / depth)


class MoveStats:
    __slots__ = ("time_s", "nodes", "prunes")

    def __init__(self, time_s: float, nodes: int, prunes: int) -> None:
        self.time_s = time_s
        self.nodes = nodes
        self.prunes = prunes


def _summarise(stats: list[MoveStats]) -> dict:
    times  = [s.time_s for s in stats]
    nodes  = [s.nodes  for s in stats]
    prunes = [s.prunes for s in stats]
    tn, tp = sum(nodes), sum(prunes)
    return {
        "moves":        len(stats),
        "total_nodes":  tn,
        "avg_nodes":    _avg([float(n) for n in nodes]),
        "max_nodes":    max(nodes) if nodes else 0,
        "total_time":   sum(times),
        "avg_time":     _avg(times),
        "max_time":     max(times) if times else 0.0,
        "total_prunes": tp,
        "pruning_rate": tp / (tn + tp) if (tn + tp) > 0 else 0.0,
    }


def _write_csv_row(csv_path: str, row: dict) -> None:
    write_header = not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# AI move helpers
# ---------------------------------------------------------------------------

def _ai_move_alphabeta(game: Game, player: int, depth: int) -> Optional[tuple[int, int]]:
    reset_counters()
    _, move = alphabeta(
        game.board,
        depth=depth,
        alpha=float("-inf"),
        beta=float("inf"),
        maximizing_player=True,
        player=player,
    )
    return move


def _ai_move_minimax(game: Game, player: int, depth: int) -> Optional[tuple[int, int]]:
    reset_counters()
    _, move = minimax(
        game.board,
        depth=depth,
        maximizing_player=True,
        player=player,
    )
    return move


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_one_game(
    depth: int,
    mm_is_black: bool,
    seed: Optional[int] = None,
    max_moves: int = 225,
) -> tuple[int, list[MoveStats], list[MoveStats]]:
    """Play one game.  One side uses minimax, the other alpha-beta.

    Returns (winner, mm_stats, ab_stats).
    *mm_is_black*: if True, minimax plays Black (first mover).
    *seed*: when set, Black's first stone is placed randomly from the
            centre 5×5 region so each game starts from a distinct position.
    """
    reset_game_state()
    game = Game()
    mm_stats: list[MoveStats] = []
    ab_stats: list[MoveStats] = []

    # Random opening move (attributed to whichever colour moves first).
    if seed is not None:
        rng = random.Random(seed)
        half = BOARD_SIZE // 2
        pool = [(half + dr, half + dc) for dr in range(-2, 3) for dc in range(-2, 3)]
        r, c = rng.choice(pool)
        game.make_move(r, c)
        # First move is not counted in stats — it was not searched.

    for _ in range(max_moves):
        if game.is_over():
            break
        player = game.current_player
        is_mm_turn = (player == BLACK) == mm_is_black

        t0 = time.perf_counter()
        if is_mm_turn:
            move = _ai_move_minimax(game, player, depth)
        else:
            move = _ai_move_alphabeta(game, player, depth)
        elapsed = time.perf_counter() - t0

        stat = MoveStats(elapsed, get_node_count(), get_prune_count())
        if is_mm_turn:
            mm_stats.append(stat)
        else:
            ab_stats.append(stat)

        if move is None:
            break
        game.make_move(*move)

    return game.get_winner(), mm_stats, ab_stats


# ---------------------------------------------------------------------------
# Per-depth runner
# ---------------------------------------------------------------------------

def run_depth(
    depth: int,
    num_games: int,
    log_lines: list[str],
    run_id: str,
    csv_path: str,
    weights_json: str,
) -> dict:
    header = f"=== Depth={depth}  Minimax vs Alpha-Beta  games={num_games} ==="
    log_lines.append(header)
    print(header)

    results = {BLACK: 0, WHITE: 0, EMPTY: 0}
    all_mm_stats: list[MoveStats] = []
    all_ab_stats: list[MoveStats] = []

    for i in range(1, num_games + 1):
        # Alternate which colour minimax plays so neither colour is always favoured.
        mm_is_black = (i % 2 == 1)
        mm_role = "Black" if mm_is_black else "White"

        t0 = time.perf_counter()
        winner, mm_stats, ab_stats = play_one_game(depth, mm_is_black, seed=i)
        game_time = time.perf_counter() - t0

        results[winner] += 1
        all_mm_stats.extend(mm_stats)
        all_ab_stats.extend(ab_stats)

        mm_s = _summarise(mm_stats)
        ab_s = _summarise(ab_stats)
        total_moves = mm_s["moves"] + ab_s["moves"]

        # Node reduction this game
        node_red = (
            (mm_s["total_nodes"] - ab_s["total_nodes"]) / mm_s["total_nodes"] * 100
            if mm_s["total_nodes"] > 0 else 0.0
        )
        time_red = (
            (mm_s["total_time"] - ab_s["total_time"]) / mm_s["total_time"] * 100
            if mm_s["total_time"] > 0 else 0.0
        )

        line = (
            f"  Game {i:>3}: mm={mm_role:>5}  winner={_player_label(winner):>5}  "
            f"moves={total_moves:>3}  time={game_time:.2f}s  "
            f"mm_nodes={mm_s['total_nodes']:.0f}  ab_nodes={ab_s['total_nodes']:.0f}  "
            f"node_reduction={node_red:+.1f}%"
        )
        log_lines.append(line)
        print(line)

        _write_csv_row(csv_path, {
            "run_id":            run_id,
            "depth":             depth,
            "game_num":          i,
            "mm_role":           mm_role,
            "winner":            _player_label(winner),
            "total_moves":       total_moves,
            "game_time_s":       f"{game_time:.4f}",
            "mm_moves":          mm_s["moves"],
            "mm_total_nodes":    int(mm_s["total_nodes"]),
            "mm_avg_nodes":      f"{mm_s['avg_nodes']:.0f}",
            "mm_max_nodes":      int(mm_s["max_nodes"]),
            "mm_total_time":     f"{mm_s['total_time']:.4f}",
            "mm_avg_time":       f"{mm_s['avg_time']:.4f}",
            "mm_max_time":       f"{mm_s['max_time']:.4f}",
            "mm_pruning_rate":   "0.0000",
            "mm_avg_ebf":        f"{_ebf(mm_s['total_nodes'], mm_s['moves'], depth):.2f}",
            "ab_moves":          ab_s["moves"],
            "ab_total_nodes":    int(ab_s["total_nodes"]),
            "ab_avg_nodes":      f"{ab_s['avg_nodes']:.0f}",
            "ab_max_nodes":      int(ab_s["max_nodes"]),
            "ab_total_time":     f"{ab_s['total_time']:.4f}",
            "ab_avg_time":       f"{ab_s['avg_time']:.4f}",
            "ab_max_time":       f"{ab_s['max_time']:.4f}",
            "ab_total_prunes":   int(ab_s["total_prunes"]),
            "ab_pruning_rate":   f"{ab_s['pruning_rate']:.4f}",
            "ab_avg_ebf":        f"{_ebf(ab_s['total_nodes'], ab_s['moves'], depth):.2f}",
            "node_reduction_pct": f"{node_red:.2f}",
            "time_reduction_pct": f"{time_red:.2f}",
            "weights":           weights_json,
        })

    # Aggregated summary
    log_lines.append("")
    mm_all = _summarise(all_mm_stats)
    ab_all = _summarise(all_ab_stats)

    node_red_all = (
        (mm_all["total_nodes"] - ab_all["total_nodes"]) / mm_all["total_nodes"] * 100
        if mm_all["total_nodes"] > 0 else 0.0
    )
    time_red_all = (
        (mm_all["total_time"] - ab_all["total_time"]) / mm_all["total_time"] * 100
        if mm_all["total_time"] > 0 else 0.0
    )

    ab_wins = sum(
        1 for s in [all_ab_stats]  # placeholder — tracked via results below
    )
    summary = [
        f"  Results: Black wins={results[BLACK]}  White wins={results[WHITE]}  Draws={results[EMPTY]}",
        "",
        f"  Minimax  (depth={depth}):",
        f"    avg_nodes={mm_all['avg_nodes']:.0f}  max_nodes={mm_all['max_nodes']}  "
        f"avg_time={mm_all['avg_time']:.4f}s  pruning_rate=0.0%  "
        f"avg_EBF={_ebf(mm_all['total_nodes'], mm_all['moves'], depth):.2f}",
        "",
        f"  AlphaBeta (depth={depth}):",
        f"    avg_nodes={ab_all['avg_nodes']:.0f}  max_nodes={ab_all['max_nodes']}  "
        f"avg_time={ab_all['avg_time']:.4f}s  pruning_rate={ab_all['pruning_rate']:.1%}  "
        f"avg_EBF={_ebf(ab_all['total_nodes'], ab_all['moves'], depth):.2f}",
        "",
        f"  >>> Node reduction:  {node_red_all:+.1f}%   Time reduction: {time_red_all:+.1f}%",
        f"      Speedup factor:  {mm_all['avg_time'] / ab_all['avg_time']:.2f}x  "
        f"(minimax_avg / alphabeta_avg)"
        if ab_all["avg_time"] > 0 else "      (no alphabeta moves recorded)",
    ]
    for s in summary:
        log_lines.append(s)
        print(s)

    log_lines.append("")
    print()

    return {
        "depth":          depth,
        "mm_avg_nodes":   mm_all["avg_nodes"],
        "ab_avg_nodes":   ab_all["avg_nodes"],
        "mm_avg_time":    mm_all["avg_time"],
        "ab_avg_time":    ab_all["avg_time"],
        "ab_pruning_rate": ab_all["pruning_rate"],
        "node_red_pct":   node_red_all,
        "time_red_pct":   time_red_all,
        "speedup":        mm_all["avg_time"] / ab_all["avg_time"] if ab_all["avg_time"] > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Summary table across depths
# ---------------------------------------------------------------------------

def _print_summary_table(rows: list[dict], log_lines: list[str]) -> None:
    header = "=== Algorithm Comparison Summary ==="
    log_lines.append(header)
    print(header)

    col_h = (
        f"  {'depth':>5}  {'mm_avg_nodes':>14}  {'ab_avg_nodes':>14}  "
        f"{'node_red%':>10}  {'ab_prune%':>10}  {'speedup':>8}"
    )
    log_lines.append(col_h)
    print(col_h)

    for r in rows:
        row = (
            f"  {r['depth']:>5}  {r['mm_avg_nodes']:>14.0f}  {r['ab_avg_nodes']:>14.0f}  "
            f"{r['node_red_pct']:>9.1f}%  {r['ab_pruning_rate']:>9.1%}  "
            f"{r['speedup']:>7.2f}x"
        )
        log_lines.append(row)
        print(row)

    log_lines.append("")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gomoku algorithm comparison: Minimax vs Alpha-Beta"
    )
    p.add_argument(
        "--depths", type=int, nargs="+", default=None,
        help="Depths to test (default: 1 2 3). Ignored when --depth is set.",
    )
    p.add_argument(
        "--depth", type=int, default=None,
        help="Test a single depth only.",
    )
    p.add_argument(
        "--games", type=int, default=3,
        help="Games per depth level (default 3). Use even numbers for balanced colour assignment.",
    )
    p.add_argument("--output", type=str, default=TXT_FILE, help="Human-readable log file")
    p.add_argument("--csv",    type=str, default=CSV_FILE, help="CSV output file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reset_weights()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = args.output.replace("{run_id}", run_id)
    csv_path = args.csv.replace("{run_id}", run_id)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    weights_json = json.dumps(get_weights(), sort_keys=True)

    log_lines: list[str] = []
    banner = (
        f"Gomoku Algorithm Comparison: Minimax vs Alpha-Beta  (run_id={run_id})\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Weights: {weights_json}\n"
        f"Note: Minimax colour alternates each game to balance first-mover advantage.\n"
    )
    log_lines.append(banner)
    print(banner)

    depths = [args.depth] if args.depth is not None else (args.depths or [1, 2, 3])
    summary_rows: list[dict] = []

    for d in depths:
        row = run_depth(d, args.games, log_lines, run_id, csv_path, weights_json)
        summary_rows.append(row)

    if len(summary_rows) > 1:
        _print_summary_table(summary_rows, log_lines)

    footer = f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    log_lines.append(footer)
    print(footer)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        f.write("\n")

    print(f"\nResults written to {txt_path}")
    print(f"CSV data written to  {csv_path}")


if __name__ == "__main__":
    main()
