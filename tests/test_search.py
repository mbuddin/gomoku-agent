from __future__ import annotations

from gomoku.board import BLACK, WHITE, Board
from gomoku.search import alphabeta, minimax


def test_minimax_prefers_center_on_empty_board() -> None:
	board = Board()
	score, move = minimax(board, depth=1, maximizing_player=True, player=BLACK)
	assert move == (board.size // 2, board.size // 2)
	assert isinstance(score, int)


def test_alphabeta_finds_immediate_winning_move() -> None:
	board = Board()
	for col in range(4):
		board.place_stone(7, col, BLACK)
	score, move = alphabeta(
		board,
		depth=2,
		alpha=float("-inf"),
		beta=float("inf"),
		maximizing_player=True,
		player=BLACK,
	)
	assert move == (7, 4)
	assert score > 100_000


def test_minimax_and_alphabeta_agree_on_best_move() -> None:
	board = Board()
	board.place_stone(7, 7, BLACK)
	board.place_stone(7, 8, WHITE)
	board.place_stone(8, 7, BLACK)
	board.place_stone(8, 8, WHITE)

	_, mm_move = minimax(board, depth=2, maximizing_player=True, player=BLACK)
	_, ab_move = alphabeta(
		board,
		depth=2,
		alpha=float("-inf"),
		beta=float("inf"),
		maximizing_player=True,
		player=BLACK,
	)

	assert mm_move == ab_move
