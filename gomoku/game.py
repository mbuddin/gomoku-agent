from __future__ import annotations
from gomoku.board import Board, BLACK, WHITE, EMPTY


class Game:
    """Manages the state of a single Gomoku match.

    Responsibilities:
    - Track whose turn it is.
    - Accept and validate moves.
    - Detect game-over conditions (win or draw).
    - Maintain a move history for undo support.
    """

    def __init__(self) -> None:
        self.board: Board = Board()
        self.current_player: int = BLACK  # BLACK always moves first per Gomoku rules
        self.winner: int = EMPTY          # EMPTY means no winner yet (also used for draws)
        self.move_history: list[tuple[int, int]] = []  # stack of (row, col) for undo support
        self._game_over: bool = False      # set True on win or draw; blocks further moves

    # ------------------------------------------------------------------
    # Core actions
    # ------------------------------------------------------------------

    def make_move(self, row: int, col: int) -> bool:
        """Attempt to place the current player's stone at (row, col).

        Returns True if the move was accepted, False if it was illegal.
        After a successful move the turn advances (or the game ends).
        """
        if self._game_over:
            return False
        if not self.board.is_valid_move(row, col):  # rejects out-of-bounds or occupied cells
            return False

        self.board.place_stone(row, col, self.current_player)
        self.move_history.append((row, col))  # record before win check so undo stays consistent

        if self.board.check_win(row, col, self.current_player):
            # Only need to check the stone just placed — earlier stones were already checked
            self.winner = self.current_player
            self._game_over = True
        elif self.board.is_full():
            self._game_over = True  # draw — winner stays EMPTY
        else:
            self._switch_player()  # advance turn only when the game continues

        return True

    def undo_move(self) -> bool:
        """Undo the last move and restore the previous turn.

        Returns True if a move was undone, False if there is nothing to undo.
        """
        if not self.move_history:
            return False

        row, col = self.move_history.pop()
        self.board.remove_stone(row, col)

        # Always clear game-over and winner: the position before this move
        # may no longer be terminal (e.g., undoing a winning move).
        self._game_over = False
        self.winner = EMPTY
        # Switch back to whoever just played — their stone was just removed
        self._switch_player()

        return True

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_over(self) -> bool:
        """Return True when the game has ended (win or draw)."""
        return self._game_over

    def get_winner(self) -> int:
        """Return the winning player (BLACK or WHITE), or EMPTY for a draw
        / ongoing game."""
        return self.winner

    def reset(self) -> None:
        """Reset to a fresh game."""
        self.board.reset()
        self.current_player = BLACK
        self.winner = EMPTY
        self.move_history.clear()
        self._game_over = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _switch_player(self) -> None:
        # Toggle between the two players; called after each move and on undo
        self.current_player = WHITE if self.current_player == BLACK else BLACK

    def __repr__(self) -> str:  # pragma: no cover
        player_name = {BLACK: "Black", WHITE: "White", EMPTY: "None"}
        status = (
            f"Winner: {player_name[self.winner]}"
            if self._game_over
            else f"Turn: {player_name[self.current_player]}"
        )
        return f"{self.board!r}\n{status}"
