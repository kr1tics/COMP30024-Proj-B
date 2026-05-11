from __future__ import annotations

from .board import Board
from referee.game import Action, Coord, PlayerColor


BOARD_SIZE = 8
BOARD_CENTER = 3.5

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
    is_terminal, winner = board.is_terminal()
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


def _token_advantage(
    board: Board, color: PlayerColor, opponent: PlayerColor
) -> float:
    return float(board.total_tokens(color) - board.total_tokens(opponent))


def _height_advantage(
    board: Board, color: PlayerColor, opponent: PlayerColor
) -> float:
    return float(_max_stack_height(board, color) - _max_stack_height(board, opponent))


def _cascade_threat(
    board: Board, color: PlayerColor, opponent: PlayerColor
) -> float:
    my_stacks = board.get_stacks(color)
    enemy_coords = [coord for coord, _ in board.get_stacks(opponent)]
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
        sum(1 for coord, _ in board.get_stacks(color) if _is_near_edge(coord))
    )


def _stack_consolidation(board: Board, color: PlayerColor) -> float:
    stacks = board.get_stacks(color)
    if not stacks:
        return 0.0
    return float(max(height for _, height in stacks) - len(stacks))


def _eat_threat_count(
    board: Board, attacker: PlayerColor, defender: PlayerColor
) -> int:
    defender_stacks = board.get_stacks(defender)
    count = 0

    for coord, height in board.get_stacks(attacker):
        for enemy_coord, enemy_height in defender_stacks:
            if _is_adjacent(coord, enemy_coord) and height >= enemy_height:
                count += 1

    return count


def _max_stack_height(board: Board, color: PlayerColor) -> int:
    return max((height for _, height in board.get_stacks(color)), default=0)


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
    own_stacks = board.get_stacks(color)
    if not own_stacks:
        return 4.0

    min_dist = min(_manhattan_distance(coord, own_coord) for own_coord, _ in own_stacks)
    return float(min(min_dist, 4))


def _placement_proximity_to_enemy(
    board: Board, opponent: PlayerColor, coord: Coord
) -> float:
    enemy_stacks = board.get_stacks(opponent)
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
