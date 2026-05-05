from __future__ import annotations

from gomoku.board import BLACK, EMPTY, WHITE, Board

WIN_SCORE = 1_000_000

# ---------------------------------------------------------------------------
# Pattern-type weights
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: dict[str, int] = {
    "five": WIN_SCORE,
    "open_four": 100_000,   # both ends open, unstoppable
    "rush_four": 8_000,     # one completion point, must respond
    "open_three": 3_000,    # can form open four next move
    "sleep_three": 400,     # can only form rush four
    "open_two": 100,        # room to grow
    "sleep_two": 15,        # limited growth
}

# Active weights — starts as a copy of defaults; can be overridden at runtime.
_WEIGHTS: dict[str, int] = dict(_DEFAULT_WEIGHTS)


def get_weights() -> dict[str, int]:
    """Return a copy of the current active weights."""
    return dict(_WEIGHTS)


def set_weights(overrides: dict[str, int]) -> None:
    """Override active weights. Only keys present in *overrides* are changed."""
    for k, v in overrides.items():
        if k in _WEIGHTS:
            _WEIGHTS[k] = v


def reset_weights() -> None:
    """Restore all weights to their defaults."""
    _WEIGHTS.update(_DEFAULT_WEIGHTS)

# ---------------------------------------------------------------------------
# Pattern templates
#
# Encoding:  x = target stone, o = opponent / boundary, _ = empty.
# Each extracted line is bounded by 'o' at both ends before matching.
# ---------------------------------------------------------------------------
_PATTERN_TABLE: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Five
    ("five", ("xxxxx",)),
    # Open four: _xxxx_ — both sides open
    ("open_four", ("_xxxx_",)),
    # Rush four: exactly one way to complete five
    (
        "rush_four",
        (
            "oxxxx_", "_xxxxo",           # consecutive, one end blocked
            "xxx_x", "x_xxx", "xx_xx",   # gap four
        ),
    ),
    # Open three: can become open four in one move
    (
        "open_three",
        (
            "_xxx_",                       # consecutive open three
            "_x_xx_", "_xx_x_",           # jump three
        ),
    ),
    # Sleep three: can only become rush four
    (
        "sleep_three",
        (
            "oxxx__", "__xxxo",           # consecutive, one end fully blocked
            "ox_xx_", "_xx_xo",           # jump, one end blocked
            "oxx_x_", "_x_xxo",           # jump, one end blocked
            "x__xx", "xx__x",             # two-gap three
            "x_x_x",                      # double-gap three
        ),
    ),
    # Open two
    (
        "open_two",
        (
            "__xx__",                     # centered open two
            "_x_x_",                      # jump two, both open
        ),
    ),
    # Sleep two
    (
        "sleep_two",
        (
            "oxx___", "___xxo",           # consecutive, one end blocked
            "ox_x__", "__x_xo",           # jump two, one end blocked
            "x___x",                      # far two
        ),
    ),
)


# ---------------------------------------------------------------------------
# Line extraction
# ---------------------------------------------------------------------------
def _opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def _get_all_lines(board: Board) -> list[list[int]]:
    """Return every row, column, and diagonal of length >= 5."""
    size = board.size
    g = board.grid
    lines: list[list[int]] = []

    # Rows
    for r in range(size):
        lines.append(g[r][:])

    # Columns
    for c in range(size):
        lines.append([g[r][c] for r in range(size)])

    # Main diagonals (↘)
    for d in range(-(size - 1), size):
        line: list[int] = []
        for r in range(size):
            c = r + d
            if 0 <= c < size:
                line.append(g[r][c])
        if len(line) >= 5:
            lines.append(line)

    # Anti-diagonals (↗)
    for d in range(0, 2 * size - 1):
        line = []
        for r in range(size):
            c = d - r
            if 0 <= c < size:
                line.append(g[r][c])
        if len(line) >= 5:
            lines.append(line)

    return lines


def _encode_line(line: list[int], target: int) -> str:
    """Encode a board line as a pattern string bounded by 'o' on each end."""
    opp = _opponent(target)
    parts = ["o"]
    for cell in line:
        if cell == target:
            parts.append("x")
        elif cell == EMPTY:
            parts.append("_")
        else:
            parts.append("o")
    parts.append("o")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Counting & scoring
# ---------------------------------------------------------------------------
def _count_patterns(board: Board, target: int) -> dict[str, int]:
    """Count occurrences of each pattern type for *target* across every line."""
    counts: dict[str, int] = {name: 0 for name in _WEIGHTS}

    for line in _get_all_lines(board):
        encoded = _encode_line(line, target)
        for name, patterns in _PATTERN_TABLE:
            for pat in patterns:
                counts[name] += encoded.count(pat)

    return counts


def _score_with_bonus(counts: dict[str, int]) -> float:
    """Convert pattern counts to a score including combination-threat bonuses."""
    score = float(sum(counts[k] * _WEIGHTS[k] for k in _WEIGHTS))

    # Combined-threat bonuses
    if counts["open_four"] >= 1:
        score += 80_000                                             # open four is nearly unstoppable
    if counts["rush_four"] >= 2:
        score += 60_000                                             # double rush four
    if counts["rush_four"] >= 1 and counts["open_three"] >= 1:
        score += 50_000                                             # four-three kill
    if counts["open_three"] >= 2:
        score += 10_000                                             # double open three

    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate(board: Board, player: int) -> float:
    """Return a heuristic score for *board* from *player*'s perspective.

    Uses pattern-based evaluation across all lines of the board.
    Recognises standard Gomoku patterns: five, open/rush four,
    open/sleep three, open/sleep two, plus combined-threat bonuses
    for double-three, four-three kill, etc.
    """
    player_counts = _count_patterns(board, player)
    opponent_counts = _count_patterns(board, _opponent(player))

    player_score = _score_with_bonus(player_counts)
    opponent_score = _score_with_bonus(opponent_counts)

    return player_score - opponent_score * 1.1
