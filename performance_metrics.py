"""
performance_metrics.py
Measures the Gomoku AI agent's performance.
"""

import time
import random

from gomoku.board import BLACK, WHITE, Board
from gomoku.game import Game
from gomoku.move_gen import get_candidate_moves
from gomoku.search import alphabeta


# ── Helper: random opponent move ──────────────────────────────────────────────

def random_move(board: Board):
    candidates = get_candidate_moves(board)
    return random.choice(candidates) if candidates else None


# ── Metric 1: How long does the AI take to pick a move? ──────────────────────

def measure_decision_time():
    print("=" * 50)
    print("Metric 1: AI Decision Time per Move")
    print("=" * 50)

    # Set up a simple mid-game board
    board = Board()
    opening_moves = [
        (7, 7, BLACK), (7, 8, WHITE),
        (8, 7, BLACK), (6, 8, WHITE),
        (8, 8, BLACK), (9, 8, WHITE),
    ]
    for r, c, player in opening_moves:
        board.place_stone(r, c, player)

    for depth in [1, 2, 3]:
        times = []
        for _ in range(5):
            start = time.time()
            alphabeta(board, depth, float("-inf"), float("inf"), True, BLACK)
            end = time.time()
            times.append(end - start)

        avg = sum(times) / len(times)
        print(f"  Depth {depth}: {avg:.4f}s")

    print()


# ── Metric 2: Win rate against a random player ────────────────────────────────

def measure_win_rate(num_games=100, depth=2):
    print("=" * 50)
    print(f"Metric 2: Win Rate (AI depth={depth} vs Random, {num_games} games)")
    print("=" * 50)

    wins = 0
    losses = 0
    draws = 0

    for i in range(num_games):
        game = Game()

        while not game.is_over():
            if game.current_player == BLACK:
                # AI plays BLACK
                _, move = alphabeta(
                    game.board, depth,
                    float("-inf"), float("inf"),
                    True, BLACK
                )
            else:
                # Random plays WHITE
                move = random_move(game.board)

            if move is None:
                break
            game.make_move(*move)

        winner = game.get_winner()
        if winner == BLACK:
            wins += 1
        elif winner == WHITE:
            losses += 1
        else:
            draws += 1

    print(f"  Wins: {wins}  Losses: {losses}  Draws: {draws}")
    print(f"  Win rate: {wins / num_games * 100:.0f}%")
    print()


# ── Metric 3: Average game length ────────────────────────────────────────────

def measure_game_length(num_games=100, depth=2):
    print("=" * 50)
    print(f"Metric 3: Average Game Length ({num_games} games)")
    print("=" * 50)

    total_moves = 0

    for _ in range(num_games):
        game = Game()

        while not game.is_over():
            if game.current_player == BLACK:
                _, move = alphabeta(
                    game.board, depth,
                    float("-inf"), float("inf"),
                    True, BLACK
                )
            else:
                move = random_move(game.board)

            if move is None:
                break
            game.make_move(*move)

        total_moves += len(game.move_history)

    avg = total_moves / num_games
    print(f"  Average moves per game: {avg:.1f}")
    print()


# ── Run all metrics ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    print("\n=== GOMOKU AGENT PERFORMANCE METRICS ===\n")
    measure_decision_time()
    measure_win_rate(num_games=100, depth=2)
    measure_game_length(num_games=100, depth=2)
    print("Done!")