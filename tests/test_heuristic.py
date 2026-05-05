from __future__ import annotations

from gomoku.board import BLACK, WHITE, Board
from gomoku.heuristic import WIN_SCORE, evaluate


def test_evaluate_detects_five_in_a_row() -> None:
    board = Board()
    for col in range(5):
        board.place_stone(7, col, BLACK)

    assert evaluate(board, BLACK) >= WIN_SCORE
    assert evaluate(board, WHITE) <= -WIN_SCORE


def test_open_four_scores_higher_than_open_three() -> None:
    open_three = Board()
    for col in range(5, 8):
        open_three.place_stone(7, col, BLACK)

    open_four = Board()
    for col in range(5, 9):
        open_four.place_stone(7, col, BLACK)

    assert evaluate(open_four, BLACK) > evaluate(open_three, BLACK)


def test_opponent_threat_reduces_score() -> None:
    board = Board()
    for col in range(4, 8):
        board.place_stone(7, col, WHITE)

    assert evaluate(board, BLACK) < 0
    assert evaluate(board, WHITE) > 0


def test_gap_four_detected() -> None:
    """A gap-four pattern (X_XXX) should score higher than an open three."""
    board = Board()
    # X _ X X X  on row 7 (gap at col 6)
    board.place_stone(7, 5, BLACK)
    board.place_stone(7, 7, BLACK)
    board.place_stone(7, 8, BLACK)
    board.place_stone(7, 9, BLACK)

    simple_three = Board()
    for col in range(5, 8):
        simple_three.place_stone(7, col, BLACK)

    assert evaluate(board, BLACK) > evaluate(simple_three, BLACK)


def test_jump_three_detected() -> None:
    """A jump-three pattern (_X_XX_) should have positive score."""
    board = Board()
    # _ X _ X X _  on row 7
    board.place_stone(7, 5, BLACK)
    board.place_stone(7, 7, BLACK)
    board.place_stone(7, 8, BLACK)

    assert evaluate(board, BLACK) > 0


def test_double_open_three_bonus() -> None:
    """Two independent open threes should score more than twice one."""
    single = Board()
    for col in range(5, 8):
        single.place_stone(7, col, BLACK)

    double = Board()
    # Horizontal three
    for col in range(5, 8):
        double.place_stone(7, col, BLACK)
    # Vertical three far away
    for row in range(2, 5):
        double.place_stone(row, 12, BLACK)

    assert evaluate(double, BLACK) > 2 * evaluate(single, BLACK)
