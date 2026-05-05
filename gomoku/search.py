from __future__ import annotations
from typing import Optional
from gomoku.board import BLACK, WHITE, BOARD_SIZE, Board
from gomoku.heuristic import evaluate
from gomoku.move_gen import get_candidate_moves

# Score returned when a winning position is found.
WIN_SCORE = 1_000_000.0
# Maximum number of candidate moves evaluated per search node.
MAX_CANDIDATES = 15

# ---------------------------------------------------------------------------
# Transposition table
# ---------------------------------------------------------------------------
_TT_EXACT = 0   # stored score is the true minimax value
_TT_LOWER = 1   # true score >= stored (beta cutoff; maximizer)
_TT_UPPER = 2   # true score <= stored (all-moves-fail-low; minimizer cutoff)

# _tt: position_hash -> (flag, depth, score, best_move)
_tt: dict[int, tuple[int, int, float, Optional[tuple[int, int]]]] = {}

# ---------------------------------------------------------------------------
# Killer-move table  (2 killers per remaining-depth level)
# ---------------------------------------------------------------------------
_MAX_KILLER_DEPTH = 20
_killers: list[list[Optional[tuple[int, int]]]] = [
    [None, None] for _ in range(_MAX_KILLER_DEPTH + 1)
]

# ---------------------------------------------------------------------------
# History heuristic  (accumulated cutoff credit per board square)
# ---------------------------------------------------------------------------
_history: list[list[int]] = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]

# ---------------------------------------------------------------------------
# Move-counter / prune-counter (reset before each top-level search call)
# ---------------------------------------------------------------------------
_node_counter: int = 0
_prune_counter: int = 0


def get_node_count() -> int:
    """Return the number of nodes explored in the last search call."""
    return _node_counter


def get_prune_count() -> int:
    """Return the number of alpha-beta prune events in the last search call."""
    return _prune_counter


def reset_counters() -> None:
    """Reset node and prune counters to zero (call before each top-level search)."""
    global _node_counter, _prune_counter
    _node_counter = 0
    _prune_counter = 0


def reset_game_state() -> None:
    """Clear the transposition table, killer moves, and history heuristic.

    Call this at the start of each new game so stale data from a previous
    game does not pollute the search tables.
    """
    _tt.clear()
    for slot in _killers:
        slot[0] = None
        slot[1] = None
    for row in _history:
        for i in range(len(row)):
            row[i] = 0


def _update_killer(depth: int, move: tuple[int, int]) -> None:
    """Promote *move* to the front of the killer list for *depth*."""
    if depth > _MAX_KILLER_DEPTH:
        return
    slot = _killers[depth]
    if move != slot[0]:
        slot[1] = slot[0]
        slot[0] = move


def _opponent(player: int) -> int:
    """Return the opponent of *player*."""
    return WHITE if player == BLACK else BLACK


# ---------------------------------------------------------------------------
# Move ordering – improves alpha-beta pruning efficiency
# ---------------------------------------------------------------------------
def _order_moves(
    board: Board,
    candidates: list[tuple[int, int]],
    current: int,
    depth: int,
    tt_move: Optional[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Sort candidates so the most promising moves are examined first.

    Order:
      1. Immediate wins  (current player completes five)
      2. Blocks          (prevents opponent from completing five)
      3. TT move         (best move from a previous search of this position)
      4. Killer moves    (non-tactical moves that caused beta-cutoffs at this depth)
      5. Rest            (sorted descending by history-heuristic score)

    Better ordering causes alpha-beta to prune far more branches.
    """
    opp = _opponent(current)
    winning:  list[tuple[int, int]] = []
    blocking: list[tuple[int, int]] = []
    tt_list:  list[tuple[int, int]] = []
    killer_list: list[tuple[int, int]] = []
    rest:     list[tuple[int, int]] = []

    # Collect killer moves for this depth level.
    depth_killers: set[tuple[int, int]] = set()
    if 0 <= depth <= _MAX_KILLER_DEPTH:
        for km in _killers[depth]:
            if km is not None:
                depth_killers.add(km)

    for row, col in candidates:
        # 1. Check if this move wins immediately for the current player.
        board.place_stone(row, col, current)
        if board.check_win(row, col, current):
            board.remove_stone(row, col)
            winning.append((row, col))
            continue
        board.remove_stone(row, col)

        # 2. Check if this move blocks an immediate win for the opponent.
        board.place_stone(row, col, opp)
        if board.check_win(row, col, opp):
            board.remove_stone(row, col)
            blocking.append((row, col))
            continue
        board.remove_stone(row, col)

        # 3. TT move (best move from a prior search of this exact position).
        if (row, col) == tt_move:
            tt_list.append((row, col))
        # 4. Killer move for this depth.
        elif (row, col) in depth_killers:
            killer_list.append((row, col))
        else:
            rest.append((row, col))

    # Sort the remaining moves by accumulated history score (descending).
    rest.sort(key=lambda m: _history[m[0]][m[1]], reverse=True)

    return winning + blocking + tt_list + killer_list + rest


# ---------------------------------------------------------------------------
# Minimax (plain, no pruning)
# ---------------------------------------------------------------------------
def minimax(
    board: Board,
    depth: int,
    maximizing_player: bool,
    player: int,
) -> tuple[int, Optional[tuple[int, int]]]:
    """Depth-limited minimax search without alpha-beta pruning.

    Returns (score, best_move).  *player* is the AI whose perspective
    we score from; *maximizing_player* indicates whose turn it is in
    the current node.
    """
    global _node_counter
    _node_counter += 1

    # Leaf node: evaluate the board position with the heuristic.
    if depth == 0 or board.is_full():
        return int(evaluate(board, player)), None

    # Determine who is moving at this node.
    current = player if maximizing_player else _opponent(player)
    candidates = get_candidate_moves(board)
    if not candidates:
        return int(evaluate(board, player)), None

    # Order moves and cap the branching factor.
    candidates = _order_moves(board, candidates, current, depth, None)[:MAX_CANDIDATES]

    best_move: Optional[tuple[int, int]] = None
    if maximizing_player:
        # AI's turn: pick the move with the highest score.
        best_score = float("-inf")
        for row, col in candidates:
            board.place_stone(row, col, current)
            # Early termination: if this move wins, return immediately.
            if board.check_win(row, col, current):
                board.remove_stone(row, col)
                return int(WIN_SCORE + depth), (row, col)
            # Recurse with the opponent as the minimizing player.
            score, _ = minimax(board, depth - 1, False, player)
            board.remove_stone(row, col)
            if score > best_score:
                best_score = float(score)
                best_move = (row, col)
        return int(best_score), best_move

    # Opponent's turn: pick the move with the lowest score (worst for AI).
    best_score = float("inf")
    for row, col in candidates:
        board.place_stone(row, col, current)
        # Early termination: opponent wins.
        if board.check_win(row, col, current):
            board.remove_stone(row, col)
            return int(-WIN_SCORE - depth), (row, col)
        # Recurse with the AI as the maximizing player.
        score, _ = minimax(board, depth - 1, True, player)
        board.remove_stone(row, col)
        if score < best_score:
            best_score = float(score)
            best_move = (row, col)
    return int(best_score), best_move


# ---------------------------------------------------------------------------
# Alpha-Beta pruning (optimised minimax)
# ---------------------------------------------------------------------------
def alphabeta(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    player: int,
) -> tuple[float, Optional[tuple[int, int]]]:
    """Minimax search with alpha-beta pruning, transposition table,
    killer-move heuristic, and history heuristic.

    *alpha* - the best score the maximizer can guarantee so far.
    *beta*  - the best score the minimizer can guarantee so far.
    When beta <= alpha the remaining branches cannot affect the result
    and are pruned (skipped).

    Returns (score, best_move).
    """
    global _node_counter, _prune_counter
    _node_counter += 1

    # ── Transposition-table lookup ────────────────────────────────────
    tt_key = board.hash
    tt_entry = _tt.get(tt_key)
    tt_move: Optional[tuple[int, int]] = None
    if tt_entry is not None:
        tt_flag, tt_depth, tt_score, tt_move = tt_entry
        if tt_depth >= depth:
            if tt_flag == _TT_EXACT:
                return tt_score, tt_move
            elif tt_flag == _TT_LOWER and tt_score >= beta:
                return tt_score, tt_move
            elif tt_flag == _TT_UPPER and tt_score <= alpha:
                return tt_score, tt_move
        # tt_move is still useful for ordering even when we can't use the score.

    # ── Leaf node ─────────────────────────────────────────────────────
    if depth == 0 or board.is_full():
        score = evaluate(board, player)
        _tt[tt_key] = (_TT_EXACT, 0, score, None)
        return score, None

    # ── Move generation & ordering ────────────────────────────────────
    current = player if maximizing_player else _opponent(player)
    candidates = get_candidate_moves(board)
    if not candidates:
        score = evaluate(board, player)
        return score, None

    candidates = _order_moves(board, candidates, current, depth, tt_move)[:MAX_CANDIDATES]

    orig_alpha = alpha
    orig_beta  = beta
    best_move: Optional[tuple[int, int]] = None

    if maximizing_player:
        # ── Maximizing ────────────────────────────────────────────────
        best_score = float("-inf")
        for row, col in candidates:
            board.place_stone(row, col, current)
            # Immediate win – no need to search deeper.
            if board.check_win(row, col, current):
                board.remove_stone(row, col)
                return WIN_SCORE + depth, (row, col)
            score, _ = alphabeta(board, depth - 1, alpha, beta, False, player)
            board.remove_stone(row, col)

            if score > best_score:
                best_score = score
                best_move = (row, col)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                _prune_counter += 1
                _update_killer(depth, (row, col))
                _history[row][col] += depth * depth
                break

        # Store in TT.
        if best_score >= beta:
            _tt[tt_key] = (_TT_LOWER, depth, best_score, best_move)
        elif best_score <= orig_alpha:
            _tt[tt_key] = (_TT_UPPER, depth, best_score, best_move)
        else:
            _tt[tt_key] = (_TT_EXACT, depth, best_score, best_move)
        return best_score, best_move

    # ── Minimizing ────────────────────────────────────────────────────
    best_score = float("inf")
    for row, col in candidates:
        board.place_stone(row, col, current)
        # Immediate opponent win.
        if board.check_win(row, col, current):
            board.remove_stone(row, col)
            return -WIN_SCORE - depth, (row, col)
        score, _ = alphabeta(board, depth - 1, alpha, beta, True, player)
        board.remove_stone(row, col)

        if score < best_score:
            best_score = score
            best_move = (row, col)
        beta = min(beta, best_score)
        if beta <= alpha:
            _prune_counter += 1
            _update_killer(depth, (row, col))
            _history[row][col] += depth * depth
            break

    # Store in TT (minimizer perspective: cutoff = UPPER, fail-high = LOWER).
    if best_score <= alpha:
        _tt[tt_key] = (_TT_UPPER, depth, best_score, best_move)
    elif best_score >= orig_beta:
        _tt[tt_key] = (_TT_LOWER, depth, best_score, best_move)
    else:
        _tt[tt_key] = (_TT_EXACT, depth, best_score, best_move)
    return best_score, best_move
