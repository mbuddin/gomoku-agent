from __future__ import annotations
from gomoku.board import Board, EMPTY


def get_candidate_moves(
    board: Board,
    radius: int = 2,
) -> list[tuple[int, int]]:
    """Return candidate moves within *radius* of existing stones.

    Parameters
    ----------
    board:
        Current board state.
    radius:
        Manhattan neighbourhood radius around existing stones to include.
    """
    occupied_cells = [
        (row, col)
        for row in range(board.size)
        for col in range(board.size)
        if board.grid[row][col] != EMPTY
    ]

    if not occupied_cells:
        center = board.size // 2
        return [(center, center)]

    candidates: set[tuple[int, int]] = set()
    for row, col in occupied_cells:
        for row_offset in range(-radius, radius + 1):
            for col_offset in range(-radius, radius + 1):
                next_row = row + row_offset
                next_col = col + col_offset
                if board.is_valid_move(next_row, next_col):
                    candidates.add((next_row, next_col))

    if not candidates:
        return board.get_empty_cells()

    center = board.size // 2
    return sorted(
        candidates,
        key=lambda move: (abs(move[0] - center) + abs(move[1] - center), move[0], move[1]),
    )
