from __future__ import annotations

from .board import Board
from referee.game import (
    Action,
    CascadeAction,
    CARDINAL_DIRECTIONS,
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
OWN_COLOUR = PlayerColor.RED
OPPONENT_COLOUR = PlayerColor.BLUE

W_TOKEN = 12.0
W_HEIGHT_ADV = 5.0
W_CASCADE = 7.0
W_EAT = 5.0
W_EDGE = -3.0
W_SOFT_EDGE = -30.0
W_CONSOL = 2.0
W_ENEMY_STACK_COUNT = -6.0
W_ENDGAME_TOKEN = 12.0
W_ENDGAME_STACK = 10.0
W_HIGH_EDGE = -4.0
W_HIGH_CORNER = -10.0
W_MOBILITY = 1.5
W_EAT_CLUSTER = 8.0
EAT_LOOKAHEAD_PENALTY = W_TOKEN
CASCADE_LOOKAHEAD_PENALTY = EAT_LOOKAHEAD_PENALTY

W_PLACE_CENTRALITY = 4.0
W_PLACE_SPREAD = 3.0
W_PLACE_PROXIMITY = -2.0
W_PLACE_CORNER = -3.0

TERMINAL_WIN = 1e9


def evaluate(board: Board, color: PlayerColor) -> float:
    """Return a heuristic score from color's perspective."""
    apply_lookahead = color == OWN_COLOUR
    is_terminal, winner = _is_terminal(board)
    if is_terminal:
        if winner is None:
            return 0.0
        if winner == OWN_COLOUR:
            return TERMINAL_WIN
        return -TERMINAL_WIN

    color = OWN_COLOUR
    opponent = OPPONENT_COLOUR

    token_adv = _token_advantage(board, color, opponent)
    height_adv = _height_advantage(board, color, opponent)
    cascade = _cascade_threat(board, color, opponent)
    eat = _eat_threat(board, color, opponent)
    edge = _edge_danger(board, color)
    soft_edge = _edge_danger(board, color)
    consol = _stack_consolidation(board, color)
    enemy_stack_count = _stack_count(board, opponent)
    high_edge, high_corner = _high_stack_edge_corner_penalty(board, color)
    mobility = _mobility_score(board, color) - _mobility_score(board, opponent)
    eat_cluster = _eat_cluster_score(board, color) - _eat_cluster_score(board, opponent)
    eat_lookahead_danger = (
        apply_lookahead and _best_opponent_eat_token_count(board, color) is not None
    )
    cascade_lookahead_loss = (
        _best_opponent_cascade_loss(board)[0] if apply_lookahead else 0
    )

    total = (
        W_TOKEN * token_adv
        + W_HEIGHT_ADV * height_adv
        + W_CASCADE * cascade
        + W_EAT * eat
        + W_EDGE * edge
        + W_SOFT_EDGE * soft_edge
        + W_CONSOL * consol
        + W_ENEMY_STACK_COUNT * enemy_stack_count
        + W_HIGH_EDGE * high_edge
        + W_HIGH_CORNER * high_corner
        + W_MOBILITY * mobility
        + W_EAT_CLUSTER * eat_cluster
    )
    if eat_lookahead_danger:
        total -= EAT_LOOKAHEAD_PENALTY  # one-ply eat lookahead: penalise any position where opponent can eat own next turn
    if cascade_lookahead_loss:
        total -= CASCADE_LOOKAHEAD_PENALTY * cascade_lookahead_loss

    opponent_tokens = _total_tokens(board, opponent)
    if opponent_tokens <= 4:
        total += W_ENDGAME_TOKEN * (4 - opponent_tokens)
        total += W_ENDGAME_STACK * (4 - enemy_stack_count)

    return float(total)


def evaluate_placement(
    board: Board, color: PlayerColor, candidate_coord: Coord
) -> float:
    """Return a heuristic score for placing at candidate_coord."""
    color = OWN_COLOUR
    opponent = OPPONENT_COLOUR

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


def _stack_count(board: Board, color: PlayerColor) -> int:
    return len(_get_stacks(board, color))


def _high_stack_edge_corner_penalty(
    board: Board, color: PlayerColor
) -> tuple[float, float]:
    edge_penalty = 0.0
    corner_penalty = 0.0

    for coord, height in _get_stacks(board, color):
        if height < 4:
            continue
        if _is_corner(coord):
            corner_penalty += height * 2.0
        elif _is_near_edge(coord):
            edge_penalty += height

    return edge_penalty, corner_penalty


def _mobility_score(board: Board, color: PlayerColor) -> float:
    score = 0.0

    for coord, height in _get_stacks(board, color):
        legal = 0
        capture = 0

        for direction in CARDINAL_DIRECTIONS:
            dest = _next_coord(coord, direction)
            if dest is None:
                continue

            target = _stack_at(board, dest)
            if target is None or target[0] == color:
                legal += 1
            elif target[0] == color.opponent and height >= target[1]:
                capture += 1

        if height >= 4:
            score += legal * 1.5 + capture * 4.0
        else:
            score += legal + capture * 3.0

    return score


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


def _eat_cluster_score(board: Board, color: PlayerColor) -> float:
    eat_count = _legal_eat_count(board, color)
    if eat_count < 2:
        return 0.0
    return float(eat_count - 1)


def _legal_eat_count(board: Board, color: PlayerColor) -> int:
    count = 0

    for coord, height in _get_stacks(board, color):
        for direction in CARDINAL_DIRECTIONS:
            dest = _next_coord(coord, direction)
            if dest is None:
                continue

            target = _stack_at(board, dest)
            if (
                target is not None
                and target[0] == color.opponent
                and height >= target[1]
            ):
                count += 1

    return count


def _best_opponent_eat_token_count(board: Board, color: PlayerColor) -> int | None:
    best_token_count = None

    for action in _legal_eat_actions(board, OPPONENT_COLOUR):
        resulting_state = _state_after_eat(board, action)
        token_count = _token_count_in_state(resulting_state, OWN_COLOUR)
        if best_token_count is None or token_count < best_token_count:
            best_token_count = token_count

    return best_token_count


def _legal_eat_actions(board: Board, color: PlayerColor) -> list[EatAction]:
    actions = []

    for coord, height in _get_stacks(board, color):
        for direction in CARDINAL_DIRECTIONS:
            dest = _next_coord(coord, direction)
            if dest is None:
                continue

            target = _stack_at(board, dest)
            if (
                target is not None
                and target[0] == color.opponent
                and height >= target[1]
            ):
                actions.append(EatAction(coord, direction))

    return actions


def _state_after_eat(
    board: Board, action: EatAction
) -> dict[Coord, tuple[PlayerColor, int]]:
    state = _tuple_state(board)
    attacker = state.pop(action.coord, None)
    dest = _next_coord(action.coord, action.direction)
    if attacker is not None and dest is not None:
        state[dest] = attacker
    return state


def _best_opponent_cascade_loss(board: Board) -> tuple[int, int]:
    best_removed = 0
    best_advantage = 0
    before_own = _total_tokens(board, OWN_COLOUR)

    for coord, height in _get_stacks(board, OPPONENT_COLOUR):
        if height < 2:
            continue

        for direction in CARDINAL_DIRECTIONS:
            resulting_state = _state_after_cascade(board, coord, direction)
            own_removed = before_own - _token_count_in_state(
                resulting_state, OWN_COLOUR
            )
            if own_removed <= 0:
                continue

            opponent_advantage = _token_count_in_state(
                resulting_state, OPPONENT_COLOUR
            ) - _token_count_in_state(resulting_state, OWN_COLOUR)
            if (
                own_removed > best_removed
                or own_removed == best_removed
                and opponent_advantage > best_advantage
            ):
                best_removed = own_removed
                best_advantage = opponent_advantage

    return best_removed, best_advantage


def _state_after_cascade(
    board: Board, coord: Coord, direction: Direction
) -> dict[Coord, tuple[PlayerColor, int]]:
    state = _tuple_state(board)
    source = state.pop(coord, None)
    if source is None:
        return state

    _, height = source
    for steps in range(1, height + 1):
        target = _next_coord(coord, direction, steps)
        if target is None:
            continue

        if target in state:
            _push_stack_in_state(state, target, direction)
        state[target] = (OPPONENT_COLOUR, 1)

    return state


def _push_stack_in_state(
    state: dict[Coord, tuple[PlayerColor, int]], coord: Coord, direction: Direction
) -> None:
    stack = state.get(coord)
    if stack is None:
        return

    dest = _next_coord(coord, direction)
    if dest is None:
        state.pop(coord, None)
        return

    if dest in state:
        _push_stack_in_state(state, dest, direction)

    state[dest] = stack
    state.pop(coord, None)


def _tuple_state(board: Board) -> dict[Coord, tuple[PlayerColor, int]]:
    return {
        coord: (stack_color, height)
        for stack_color in (OWN_COLOUR, OPPONENT_COLOUR)
        for coord, height in _get_stacks(board, stack_color)
    }


def _token_count_in_state(
    state: dict[Coord, tuple[PlayerColor, int]], color: PlayerColor
) -> int:
    return sum(height for stack_color, height in state.values() if stack_color == color)


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
