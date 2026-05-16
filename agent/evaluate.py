from __future__ import annotations

from .board import Board
from referee.game import (
    Action,
    CascadeAction,
    Coord,
    Direction,
    EatAction,
    MoveAction,
    PlaceAction,
    PlayerColor,
)


def _in_bounds(coord: Coord) -> bool:
    return 0 <= coord.r <= 7 and 0 <= coord.c <= 7


def _get_stacks(board, color) -> list[tuple]:
    """Return [(coord, height)] for color. Works with referee's Board and agent's Board."""
    if hasattr(board, '_state'):
        return [(coord, cell.height) for coord, cell in board._state.items()
                if cell.color == color]
    return board.get_stacks(color)


def _total_tokens(board, color) -> int:
    """Return total token count for color. Works with referee's Board and agent's Board."""
    if hasattr(board, '_state'):
        return sum(cell.height for cell in board._state.values() if cell.color == color)
    return board.total_tokens(color)


def _is_terminal(board) -> tuple[bool, any]:
    """Return (is_terminal, winner). Works with referee's Board and agent's Board."""
    if hasattr(board, 'game_over'):
        return board.game_over, board.winner_color
    return board.is_terminal()


def _step(coord: Coord, direction: Direction, steps: int = 1) -> Coord:
    # Returns coord offset by `steps` in `direction`. May be out of bounds.
    offsets = {
        Direction.Up: (-1, 0),
        Direction.Down: (1, 0),
        Direction.Left: (0, -1),
        Direction.Right: (0, 1),
    }
    dr, dc = offsets[direction]
    return Coord(coord.r + dr * steps, coord.c + dc * steps)


BOARD_SIZE = 8
BOARD_CENTER = 3.5
MAX_USEFUL_MERGE = 8

W_TOKEN = 10.0
W_HEIGHT_ADV = 5.0
W_CASCADE = 8.0
W_EAT = 6.0
W_EDGE = -3.0
W_CONSOL = 2.0

W_PLACE_CENTRALITY = 4.0
W_PLACE_SPREAD = 3.0
W_PLACE_PROXIMITY = -2.0
W_PLACE_CORNER = -3.0

TERMINAL_WIN = 1e9


def evaluate(board: Board, color: PlayerColor) -> float:
    """Return a heuristic score from color's perspective."""
    is_terminal, winner = _is_terminal(board)
    if is_terminal:
        if winner is None:
            return 0.0
        if winner == color:
            return TERMINAL_WIN
        return -TERMINAL_WIN

    opponent = color.opponent

    token_adv = _token_advantage(board, color, opponent)
    height_adv = _height_advantage(board, color, opponent)
    cascade = _cascade_threat(board, color, opponent)
    eat = _eat_threat(board, color, opponent)
    edge = _edge_danger(board, color)
    consol = _stack_consolidation(board, color)

    total = (
        W_TOKEN * token_adv
        + W_HEIGHT_ADV * height_adv
        + W_CASCADE * cascade
        + W_EAT * eat
        + W_EDGE * edge
        + W_CONSOL * consol
    )
    return float(total)


def evaluate_placement(
    board: Board, color: PlayerColor, candidate_coord: Coord
) -> float:
    """Return a heuristic score for placing at candidate_coord."""
    opponent = color.opponent

    centrality = _placement_centrality(candidate_coord)
    spread = _placement_spread(board, color, candidate_coord)
    proximity = _placement_proximity_to_enemy(board, opponent, candidate_coord)
    corner = _placement_corner_avoid(candidate_coord)

    total = (
        W_PLACE_CENTRALITY * centrality
        + W_PLACE_SPREAD * spread
        + W_PLACE_PROXIMITY * proximity
        + W_PLACE_CORNER * corner
    )
    return float(total)


def action_ev(board: Board, color: PlayerColor, action: Action) -> float:
    """Estimate an action's immediate expected value without tree search."""
    new_board = board.copy()
    new_board.apply_action(color, action)
    return float(evaluate(new_board, color) - evaluate(board, color))


def rank_actions(
    board: Board, color: PlayerColor, actions: list[Action]
) -> list[tuple[float, Action]]:
    """Return actions sorted by descending immediate EV."""
    sorted_pairs = sorted(
        [(action_ev(board, color, action), action) for action in actions],
        key=lambda pair: pair[0],
        reverse=True,
    )
    return sorted_pairs


def filter_actions(board: Board, color: PlayerColor, actions: list[Action]) -> list[Action]:
    """
    Domain-specific pruning. Call before rank_actions in the search move generator.
    Never returns empty list -- falls back to highest EV action if all pruned.
    """
    filtered = []

    for action in actions:
        if isinstance(action, MoveAction):
            if _prune_merge(board, color, action) and _move_has_purpose(
                board, color, action
            ):
                filtered.append(action)
        elif isinstance(action, EatAction):
            if _eat_lands_safely(board, color, action):
                filtered.append(action)
        elif isinstance(action, CascadeAction):
            if _cascade_has_value(board, color, action):
                filtered.append(action)
        elif isinstance(action, PlaceAction):
            filtered.append(action)

    if filtered:
        return filtered
    if not actions:
        return []
    return [rank_actions(board, color, actions)[0][1]]


def _prune_merge(board: Board, color: PlayerColor, action: MoveAction) -> bool:
    src_stack = _stack_at(board, action.coord)
    dest = _next_coord(action.coord, action.direction)
    if src_stack is None or dest is None:
        return True

    dest_stack = _stack_at(board, dest)
    if dest_stack is None:
        return True

    src_color, src_height = src_stack
    dest_color, dest_height = dest_stack
    if src_color == color and dest_color == color:
        return src_height + dest_height <= MAX_USEFUL_MERGE
    return True


def _cascade_has_value(
    board: Board, color: PlayerColor, action: CascadeAction
) -> bool:
    src_stack = _stack_at(board, action.coord)
    if src_stack is None:
        return False

    _, height = src_stack
    for steps in range(1, height + 1):
        target = _next_coord(action.coord, action.direction, steps)
        if target is None:
            continue

        target_stack = _stack_at(board, target)
        if target_stack is None:
            continue

        target_color, _ = target_stack
        if target_color == color.opponent:
            pushed_dest = _next_coord(target, action.direction)
            if pushed_dest is None:
                return True
            return True

    return False


def _eat_lands_safely(board: Board, color: PlayerColor, action: EatAction) -> bool:
    src_stack = _stack_at(board, action.coord)
    dest = _next_coord(action.coord, action.direction)
    if src_stack is None or dest is None:
        return True

    _, attacker_height = src_stack
    if _is_near_edge(dest) and attacker_height >= 4:
        return False
    return True


def _move_has_purpose(board: Board, color: PlayerColor, action: MoveAction) -> bool:
    src_stack = _stack_at(board, action.coord)
    dest = _next_coord(action.coord, action.direction)
    if src_stack is None or dest is None:
        return True

    _, src_height = src_stack
    if src_height != 1 or _stack_at(board, dest) is not None:
        return True

    for direction in (
        Direction.Up,
        Direction.Down,
        Direction.Left,
        Direction.Right,
    ):
        neighbor = _next_coord(dest, direction)
        if neighbor is None:
            continue

        neighbor_stack = _stack_at(board, neighbor)
        if neighbor_stack is not None and neighbor_stack[0] == color.opponent:
            return True

    return False


def _next_coord(
    coord: Coord, direction: Direction, steps: int = 1
) -> Coord | None:
    r = coord.r + direction.r * steps
    c = coord.c + direction.c * steps
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        return None
    return _step(coord, direction, steps)


def _stack_at(board: Board, coord: Coord) -> tuple[PlayerColor, int] | None:
    state = getattr(board, "_state", None)
    if state is not None:
        cell = state.get(coord)
        if cell is None:
            return None
        if isinstance(cell, tuple):
            return cell
        if getattr(cell, "is_empty", False):
            return None
        return cell.color, cell.height

    stacks = getattr(board, "_stacks", None)
    if stacks is not None:
        for stack_color, stack_coords in stacks.items():
            if coord in stack_coords:
                return stack_color, stack_coords[coord]

    return None


def _token_advantage(
    board: Board, color: PlayerColor, opponent: PlayerColor
) -> float:
    return float(_total_tokens(board, color) - _total_tokens(board, opponent))


def _height_advantage(
    board: Board, color: PlayerColor, opponent: PlayerColor
) -> float:
    return float(_max_stack_height(board, color) - _max_stack_height(board, opponent))


def _cascade_threat(
    board: Board, color: PlayerColor, opponent: PlayerColor
) -> float:
    my_stacks = _get_stacks(board, color)
    enemy_coords = [coord for coord, _ in _get_stacks(board, opponent)]
    threat_count = 0

    for coord, height in my_stacks:
        if height < 2:
            continue

        for enemy_coord in enemy_coords:
            same_row = coord.r == enemy_coord.r
            same_col = coord.c == enemy_coord.c
            distance = abs(coord.r - enemy_coord.r) + abs(coord.c - enemy_coord.c)

            if (same_row or same_col) and 0 < distance <= height:
                threat_count += 1

    return float(threat_count)


def _eat_threat(
    board: Board, color: PlayerColor, opponent: PlayerColor
) -> float:
    my_threats = _eat_threat_count(board, color, opponent)
    opponent_threats = _eat_threat_count(board, opponent, color)
    return float(my_threats - opponent_threats)


def _edge_danger(board: Board, color: PlayerColor) -> float:
    return float(
        sum(1 for coord, _ in _get_stacks(board, color) if _is_near_edge(coord))
    )


def _stack_consolidation(board: Board, color: PlayerColor) -> float:
    stacks = _get_stacks(board, color)
    if not stacks:
        return 0.0
    return float(max(height for _, height in stacks) - len(stacks))


def _eat_threat_count(
    board: Board, attacker: PlayerColor, defender: PlayerColor
) -> int:
    defender_stacks = _get_stacks(board, defender)
    count = 0

    for coord, height in _get_stacks(board, attacker):
        for enemy_coord, enemy_height in defender_stacks:
            if _is_adjacent(coord, enemy_coord) and height >= enemy_height:
                count += 1

    return count


def _max_stack_height(board, color: PlayerColor) -> int:
    return max((height for _, height in _get_stacks(board, color)), default=0)


def _is_adjacent(coord_a, coord_b) -> bool:
    return abs(coord_a.r - coord_b.r) + abs(coord_a.c - coord_b.c) == 1


def _is_near_edge(coord) -> bool:
    return (
        coord.r == 0
        or coord.r == BOARD_SIZE - 1
        or coord.c == 0
        or coord.c == BOARD_SIZE - 1
    )


def _placement_centrality(coord: Coord) -> float:
    return float(7.0 - _manhattan_distance_to_center(coord))


def _placement_spread(board: Board, color: PlayerColor, coord: Coord) -> float:
    own_stacks = _get_stacks(board, color)
    if not own_stacks:
        return 4.0

    min_dist = min(_manhattan_distance(coord, own_coord) for own_coord, _ in own_stacks)
    return float(min(min_dist, 4))


def _placement_proximity_to_enemy(
    board: Board, opponent: PlayerColor, coord: Coord
) -> float:
    enemy_stacks = _get_stacks(board, opponent)
    if not enemy_stacks:
        return 0.0

    min_dist = min(
        _manhattan_distance(coord, enemy_coord) for enemy_coord, _ in enemy_stacks
    )
    if min_dist < 2:
        return float(2 - min(min_dist, 2))
    return 0.0


def _placement_corner_avoid(coord: Coord) -> float:
    return -1.0 if _is_corner(coord) else 0.0


def _manhattan_distance(coord_a, coord_b) -> float:
    return abs(coord_a.r - coord_b.r) + abs(coord_a.c - coord_b.c)


def _manhattan_distance_to_center(coord: Coord) -> float:
    return abs(coord.r - BOARD_CENTER) + abs(coord.c - BOARD_CENTER)


def _is_corner(coord: Coord) -> bool:
    edge_values = {0, BOARD_SIZE - 1}
    return coord.r in edge_values and coord.c in edge_values
