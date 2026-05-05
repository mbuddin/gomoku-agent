"""AI vs AI benchmark for Gomoku.

Conducts depth vs. win-rate tests and evaluates alpha-beta search efficiency.
Results are logged to benchmark_results.txt (human-readable) and
benchmark_results.csv (machine-readable, one row per game).

Usage:
    python benchmark.py                                         # default: depths 1-3, 3 games each
    python benchmark.py --depths 1 2 3 4 5 --games 10
    python benchmark.py --black-depth 3 --white-depth 2 --games 20
    python benchmark.py --weights open_four=120000 rush_four=10000  # override heuristic weights
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from typing import Optional

from gomoku.board import BLACK, BOARD_SIZE, EMPTY, WHITE
from gomoku.game import Game
from gomoku.heuristic import get_weights, reset_weights, set_weights
from gomoku.search import (
    alphabeta,
    get_node_count,
    get_prune_count,
    reset_counters,
    reset_game_state,
)

OUT_DIR  = "results/benchmark"
TXT_FILE = OUT_DIR + "/benchmark_{run_id}.txt"
CSV_FILE = OUT_DIR + "/benchmark_{run_id}.csv"

CSV_COLUMNS = [
    "run_id",
    "game_num",
    "black_depth",
    "white_depth",
    "winner",
    "total_moves",
    "game_time_s",
    # Black stats
    "b_moves",
    "b_total_time",
    "b_avg_time",
    "b_max_time",
    "b_total_nodes",
    "b_avg_nodes",
    "b_max_nodes",
    "b_total_prunes",
    "b_avg_prunes",
    # White stats
    "w_moves",
    "w_total_time",
    "w_avg_time",
    "w_max_time",
    "w_total_nodes",
    "w_avg_nodes",
    "w_max_nodes",
    "w_total_prunes",
    "w_avg_prunes",
    # Derived metrics
    "b_pruning_rate",
    "w_pruning_rate",
    "b_avg_ebf",
    "w_avg_ebf",
    # Weights snapshot (JSON)
    "weights",
]


# ── Helpers ──────────────────────────────────────────────────────────────

def _player_label(player: int) -> str:
    return {BLACK: "Black", WHITE: "White", EMPTY: "Draw"}.get(player, "?")


def ai_move(game: Game, player: int, depth: int) -> Optional[tuple[int, int]]:
    """Pick a move using alpha-beta search at the given *depth*."""
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


# ── Single game ──────────────────────────────────────────────────────────

class MoveStats:
    __slots__ = ("time_s", "nodes", "prunes")

    def __init__(self, time_s: float, nodes: int, prunes: int) -> None:
        self.time_s = time_s
        self.nodes = nodes
        self.prunes = prunes


def play_one_game(
    black_depth: int,
    white_depth: int,
    max_moves: int = 225,
    seed: Optional[int] = None,
) -> tuple[int, list[MoveStats], list[MoveStats]]:
    """Play a full AI-vs-AI game. Returns (winner, black_stats, white_stats).

    If *seed* is given, Black's first move is chosen randomly from the
    centre 5×5 region instead of by search.  This breaks the total
    determinism that otherwise makes every repeated game identical,
    so that multi-game matchups yield meaningful statistical variance.
    """
    reset_game_state()  # clear TT, killers, history for a clean game
    game = Game()
    black_stats: list[MoveStats] = []
    white_stats: list[MoveStats] = []

    # Random opening move so each game starts from a distinct position.
    if seed is not None:
        rng = random.Random(seed)
        half = BOARD_SIZE // 2
        opening_pool = [
            (half + dr, half + dc)
            for dr in range(-2, 3)
            for dc in range(-2, 3)
        ]
        row, col = rng.choice(opening_pool)
        game.make_move(row, col)
        # No search was performed for this move; don't pollute efficiency stats.

    for _ in range(max_moves):
        if game.is_over():
            break
        player = game.current_player
        depth = black_depth if player == BLACK else white_depth

        t0 = time.perf_counter()
        move = ai_move(game, player, depth)
        elapsed = time.perf_counter() - t0
        nodes = get_node_count()
        prunes = get_prune_count()

        stats = MoveStats(elapsed, nodes, prunes)
        if player == BLACK:
            black_stats.append(stats)
        else:
            white_stats.append(stats)

        if move is None:
            break
        game.make_move(*move)

    return game.get_winner(), black_stats, white_stats


# ── Reporting helpers ────────────────────────────────────────────────────

def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarise_stats(stats: list[MoveStats]) -> dict[str, float]:
    times = [s.time_s for s in stats]
    nodes = [s.nodes for s in stats]
    prunes = [s.prunes for s in stats]
    total_n = sum(nodes)
    total_p = sum(prunes)
    return {
        "moves": len(stats),
        "total_time": sum(times),
        "avg_time": _avg(times),
        "max_time": max(times) if times else 0.0,
        "total_nodes": total_n,
        "avg_nodes": _avg([float(n) for n in nodes]),
        "max_nodes": max(nodes) if nodes else 0,
        "total_prunes": total_p,
        "avg_prunes": _avg([float(p) for p in prunes]),
        "pruning_rate": total_p / (total_n + total_p) if (total_n + total_p) > 0 else 0.0,
    }


def _effective_branching_factor(total_nodes: float, num_moves: int, depth: int) -> float:
    """Avg nodes per move raised to 1/depth — approximates the effective branching factor."""
    if num_moves == 0 or depth <= 0 or total_nodes <= 0:
        return 0.0
    return (total_nodes / num_moves) ** (1.0 / depth)


def _write_csv_row(csv_path: str, row: dict) -> None:
    """Append a single row to the CSV file, writing the header if the file is new."""
    write_header = not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Matchup runner ───────────────────────────────────────────────────────

def run_matchup(
    black_depth: int,
    white_depth: int,
    num_games: int,
    log_lines: list[str],
    run_id: str,
    csv_path: str,
) -> dict:
    header = f"=== Matchup: Black(depth={black_depth}) vs White(depth={white_depth})  games={num_games} ==="
    log_lines.append(header)
    print(header)

    results = {BLACK: 0, WHITE: 0, EMPTY: 0}
    all_black_stats: list[MoveStats] = []
    all_white_stats: list[MoveStats] = []
    weights_json = json.dumps(get_weights(), sort_keys=True)

    for i in range(1, num_games + 1):
        t0 = time.perf_counter()
        winner, b_stats, w_stats = play_one_game(black_depth, white_depth, seed=i)
        game_time = time.perf_counter() - t0
        results[winner] += 1
        all_black_stats.extend(b_stats)
        all_white_stats.extend(w_stats)

        total_moves = len(b_stats) + len(w_stats)
        b_sum = _summarise_stats(b_stats)
        w_sum = _summarise_stats(w_stats)

        # Per-game CSV row
        _write_csv_row(csv_path, {
            "run_id": run_id,
            "game_num": i,
            "black_depth": black_depth,
            "white_depth": white_depth,
            "winner": _player_label(winner),
            "total_moves": total_moves,
            "game_time_s": f"{game_time:.4f}",
            "b_moves": b_sum["moves"],
            "b_total_time": f"{b_sum['total_time']:.4f}",
            "b_avg_time": f"{b_sum['avg_time']:.4f}",
            "b_max_time": f"{b_sum['max_time']:.4f}",
            "b_total_nodes": int(b_sum["total_nodes"]),
            "b_avg_nodes": f"{b_sum['avg_nodes']:.0f}",
            "b_max_nodes": int(b_sum["max_nodes"]),
            "b_total_prunes": int(b_sum["total_prunes"]),
            "b_avg_prunes": f"{b_sum['avg_prunes']:.1f}",
            "w_moves": w_sum["moves"],
            "w_total_time": f"{w_sum['total_time']:.4f}",
            "w_avg_time": f"{w_sum['avg_time']:.4f}",
            "w_max_time": f"{w_sum['max_time']:.4f}",
            "w_total_nodes": int(w_sum["total_nodes"]),
            "w_avg_nodes": f"{w_sum['avg_nodes']:.0f}",
            "w_max_nodes": int(w_sum["max_nodes"]),
            "w_total_prunes": int(w_sum["total_prunes"]),
            "w_avg_prunes": f"{w_sum['avg_prunes']:.1f}",
            "b_pruning_rate": f"{b_sum['pruning_rate']:.4f}",
            "w_pruning_rate": f"{w_sum['pruning_rate']:.4f}",
            "b_avg_ebf": f"{_effective_branching_factor(b_sum['total_nodes'], int(b_sum['moves']), black_depth):.2f}",
            "w_avg_ebf": f"{_effective_branching_factor(w_sum['total_nodes'], int(w_sum['moves']), white_depth):.2f}",
            "weights": weights_json,
        })

        line = (
            f"  Game {i:>3}: winner={_player_label(winner):>5}  "
            f"moves={total_moves:>3}  time={game_time:.2f}s  "
            f"b_nodes={b_sum['total_nodes']:.0f} b_prunes={b_sum['total_prunes']:.0f}  "
            f"w_nodes={w_sum['total_nodes']:.0f} w_prunes={w_sum['total_prunes']:.0f}"
        )
        log_lines.append(line)
        print(line)

    # Aggregated summary
    log_lines.append("")
    b_all = _summarise_stats(all_black_stats)
    w_all = _summarise_stats(all_white_stats)

    summary = [
        f"  Results: Black wins={results[BLACK]}  White wins={results[WHITE]}  Draws={results[EMPTY]}",
        f"  Black win-rate: {results[BLACK]/num_games*100:.1f}%   White win-rate: {results[WHITE]/num_games*100:.1f}%",
        "",
        f"  Black (depth={black_depth}) efficiency:",
        f"    total moves={b_all['moves']}  avg_time={b_all['avg_time']:.4f}s  max_time={b_all['max_time']:.4f}s",
        f"    avg_nodes={b_all['avg_nodes']:.0f}  max_nodes={b_all['max_nodes']}  total_nodes={b_all['total_nodes']:.0f}",
        f"    avg_prunes={b_all['avg_prunes']:.1f}  total_prunes={b_all['total_prunes']:.0f}  "
        f"pruning_rate={b_all['pruning_rate']:.1%}  "
        f"avg_EBF={_effective_branching_factor(b_all['total_nodes'], int(b_all['moves']), black_depth):.2f}",
        "",
        f"  White (depth={white_depth}) efficiency:",
        f"    total moves={w_all['moves']}  avg_time={w_all['avg_time']:.4f}s  max_time={w_all['max_time']:.4f}s",
        f"    avg_nodes={w_all['avg_nodes']:.0f}  max_nodes={w_all['max_nodes']}  total_nodes={w_all['total_nodes']:.0f}",
        f"    avg_prunes={w_all['avg_prunes']:.1f}  total_prunes={w_all['total_prunes']:.0f}  "
        f"pruning_rate={w_all['pruning_rate']:.1%}  "
        f"avg_EBF={_effective_branching_factor(w_all['total_nodes'], int(w_all['moves']), white_depth):.2f}",
    ]
    for s in summary:
        log_lines.append(s)
        print(s)

    log_lines.append("")
    print()

    b_ebf = _effective_branching_factor(b_all["total_nodes"], int(b_all["moves"]), black_depth)
    w_ebf = _effective_branching_factor(w_all["total_nodes"], int(w_all["moves"]), white_depth)
    return {
        "black_depth": black_depth,
        "white_depth": white_depth,
        "b_avg_nodes": b_all["avg_nodes"],
        "b_avg_time": b_all["avg_time"],
        "b_pruning_rate": b_all["pruning_rate"],
        "b_avg_ebf": b_ebf,
        "w_avg_nodes": w_all["avg_nodes"],
        "w_avg_time": w_all["avg_time"],
        "w_pruning_rate": w_all["pruning_rate"],
        "w_avg_ebf": w_ebf,
    }


# ── Matrix mode: every pair of depths ───────────────────────────────────

def run_depth_matrix(
    depths: list[int], num_games: int, log_lines: list[str],
    run_id: str, csv_path: str,
) -> None:
    scaling: list[dict] = []
    for bd in depths:
        for wd in depths:
            agg = run_matchup(bd, wd, num_games, log_lines, run_id, csv_path)
            if bd == wd:
                scaling.append(agg)

    if len(scaling) >= 2:
        header = "=== Depth Scaling Curve (same depth, Black perspective) ==="
        log_lines.append(header)
        print(header)
        col_header = f"  {'depth':>5}  {'avg_nodes':>10}  {'avg_time_s':>10}  {'pruning_rate':>12}  {'avg_EBF':>8}"
        log_lines.append(col_header)
        print(col_header)
        for s in scaling:
            row = (
                f"  {s['black_depth']:>5}  {s['b_avg_nodes']:>10.0f}  "
                f"{s['b_avg_time']:>10.4f}  {s['b_pruning_rate']:>11.1%}  "
                f"{s['b_avg_ebf']:>8.2f}"
            )
            log_lines.append(row)
            print(row)
        log_lines.append("")
        print()


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_weight_arg(raw: str) -> tuple[str, int]:
    """Parse a single 'key=value' weight override."""
    key, _, val = raw.partition("=")
    return key.strip(), int(val.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gomoku AI-vs-AI benchmark")
    p.add_argument(
        "--depths", type=int, nargs="+", default=None,
        help="List of depths to test in a full matrix (e.g. 1 2 3 4). "
             "Ignored when --black-depth / --white-depth are set.",
    )
    p.add_argument("--black-depth", type=int, default=None)
    p.add_argument("--white-depth", type=int, default=None)
    p.add_argument("--games", type=int, default=3, help="Games per matchup (default 3)")
    p.add_argument("--output", type=str, default=TXT_FILE, help="Human-readable log file")
    p.add_argument("--csv", type=str, default=CSV_FILE, help="CSV output file")
    p.add_argument(
        "--weights", nargs="+", default=None,
        help="Heuristic weight overrides, e.g. open_four=120000 rush_four=10000",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Apply weight overrides if provided
    if args.weights:
        overrides = dict(_parse_weight_arg(w) for w in args.weights)
        reset_weights()
        set_weights(overrides)
        print(f"Weight overrides applied: {overrides}")
    else:
        reset_weights()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines: list[str] = []

    # Build per-run filenames (substituting run_id unless the user overrode them).
    txt_path = args.output.replace("{run_id}", run_id)
    csv_path = args.csv.replace("{run_id}", run_id)
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    banner = (
        f"Gomoku AI-vs-AI Benchmark  (run_id={run_id})\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Weights: {json.dumps(get_weights(), sort_keys=True)}\n"
    )
    log_lines.append(banner)
    print(banner)

    single_matchup = args.black_depth is not None and args.white_depth is not None

    if single_matchup:
        run_matchup(args.black_depth, args.white_depth, args.games, log_lines, run_id, csv_path)
    else:
        depths = args.depths or [1, 2, 3]
        log_lines.append(f"Depth matrix: {depths}  games_per_matchup={args.games}\n")
        print(f"Depth matrix: {depths}  games_per_matchup={args.games}\n")
        run_depth_matrix(depths, args.games, log_lines, run_id, csv_path)

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
