from __future__ import annotations
import copy
import random as _random

BOARD_SIZE = 15

# ---------------------------------------------------------------------------
# Zobrist hashing tables  (generated once with a fixed seed for reproducibility)
# ---------------------------------------------------------------------------
_RNG = _random.Random(0x474F4D4F4B55)  # "GOMOKU" in hex
_ZOBRIST: list[list[list[int]]] = [
    [[_RNG.getrandbits(64) for _ in range(3)]   # indices 0=EMPTY(unused), 1=BLACK, 2=WHITE
     for _ in range(BOARD_SIZE)]
    for _ in range(BOARD_SIZE)
]
_ZOBRIST_TURN: list[int] = [0, _RNG.getrandbits(64), _RNG.getrandbits(64)]  # [0, BLACK_key, WHITE_key]

EMPTY = 0
BLACK = 1  # first player
WHITE = 2  # second player


class Board:
    """15x15 Gomoku board.

    The grid is a 2-D list where grid[row][col] holds EMPTY, BLACK, or WHITE.
    Row 0 is the top row; column 0 is the leftmost column.
    """

    def __init__(self) -> None:
        self.size: int = BOARD_SIZE
        self.grid: list[list[int]] = [
            [EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)
        ]
        self.stone_count: int = 0
        self._hash: int = 0  # Zobrist hash; updated incrementally

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    @property
    def hash(self) -> int:
        """Zobrist hash of the current position including side to move.

        Side to move is inferred from *stone_count* parity (BLACK moves
        when stone_count is even because BLACK always goes first).
        """
        side = BLACK if self.stone_count % 2 == 0 else WHITE
        return self._hash ^ _ZOBRIST_TURN[side]

    def reset(self) -> None:
        """Clear every cell back to EMPTY."""
        for row in range(self.size):
            for col in range(self.size):
                self.grid[row][col] = EMPTY
        self.stone_count = 0
        self._hash = 0

    def place_stone(self, row: int, col: int, player: int) -> None:
        """Place *player*'s stone at (row, col).

        Raises ValueError if the cell is already occupied or out of bounds.
        """
        if not self.in_bounds(row, col):
            raise ValueError(f"Position ({row}, {col}) is out of bounds.")
        if self.grid[row][col] != EMPTY:
            raise ValueError(f"Cell ({row}, {col}) is already occupied.")
        self.grid[row][col] = player
        self.stone_count += 1  # track count to make is_full() O(1)
        self._hash ^= _ZOBRIST[row][col][player]

    def remove_stone(self, row: int, col: int) -> None:
        """Remove the stone at (row, col) — used by search/undo to revert moves."""
        if not self.in_bounds(row, col):
            raise ValueError(f"Position ({row}, {col}) is out of bounds.")
        if self.grid[row][col] == EMPTY:
            raise ValueError(f"Cell ({row}, {col}) is already empty.")
        self._hash ^= _ZOBRIST[row][col][self.grid[row][col]]  # XOR out before clearing
        self.grid[row][col] = EMPTY
        self.stone_count -= 1

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def in_bounds(self, row: int, col: int) -> bool:
        """Return True when (row, col) lies inside the board."""
        return 0 <= row < self.size and 0 <= col < self.size

    def is_valid_move(self, row: int, col: int) -> bool:
        """Return True when (row, col) is in-bounds and currently empty."""
        return self.in_bounds(row, col) and self.grid[row][col] == EMPTY

    def is_full(self) -> bool:
        """Return True when every cell is occupied (board draw)."""
        return self.stone_count == self.size * self.size

    def get_empty_cells(self) -> list[tuple[int, int]]:
        """Return a list of (row, col) tuples for every empty cell."""
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.grid[r][c] == EMPTY
        ]

    # ------------------------------------------------------------------
    # Win-state detection
    # ------------------------------------------------------------------

    def check_win(self, row: int, col: int, player: int) -> bool:
        """Return True when *player* has 5 or more stones in a row
        passing through (row, col).

        This should be called immediately *after* placing a stone at
        (row, col) so that grid[row][col] == player.
        """
        # Check all four axes; each direction covers both halves of that line
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # start at 1 to include the stone just placed

            # Walk in the positive direction along this axis.
            r, c = row + dr, col + dc
            while self.in_bounds(r, c) and self.grid[r][c] == player:
                count += 1
                r += dr
                c += dc

            # Walk in the negative (opposite) direction along the same axis.
            r, c = row - dr, col - dc
            while self.in_bounds(r, c) and self.grid[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:  # standard Gomoku: 5 or more in a row wins
                return True

        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def copy(self) -> Board:
        """Return a deep copy of this board."""
        new_board = Board()
        new_board.grid = copy.deepcopy(self.grid)
        new_board.stone_count = self.stone_count
        new_board._hash = self._hash
        return new_board

    def __repr__(self) -> str:  # pragma: no cover
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = ["  " + " ".join(f"{c:2}" for c in range(self.size))]
        for r in range(self.size):
            row_str = " ".join(symbols[self.grid[r][c]] for c in range(self.size))
            rows.append(f"{r:2} {row_str}")
        return "\n".join(rows)
