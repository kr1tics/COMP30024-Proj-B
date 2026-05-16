# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

import math
import time

from agent.board import Board
from agent.evaluate import TERMINAL_WIN, evaluate, evaluate_placement
from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game import Board as SearchBoard


BOARD_SIZE = 8
DEFAULT_TIME_BUDGET = 2.5
MAX_SEARCH_DEPTH = 20


class _SearchTimeout(Exception):
    pass


def get_all_actions(board: SearchBoard, color: PlayerColor) -> list[Action]:
    """Return all legal actions for color on the given referee Board."""
    actions = []
    opponent = color.opponent

    for coord, cell in board._state.items():
        if cell.color != color:
            continue

        for direction in CARDINAL_DIRECTIONS:
            try:
                dest = coord + direction
            except ValueError:
                continue  # off board

            dest_cell = board._state[dest]

            if dest_cell.is_empty or dest_cell.color == color:
                actions.append(MoveAction(coord, direction))
            elif dest_cell.color == opponent and cell.height >= dest_cell.height:
                actions.append(EatAction(coord, direction))

        if cell.height >= 2:
            for direction in CARDINAL_DIRECTIONS:
                actions.append(CascadeAction(coord, direction))

    return actions


def find_best_action(
    board: SearchBoard,
    color: PlayerColor,
    depth: int = MAX_SEARCH_DEPTH,
    time_budget: float = DEFAULT_TIME_BUDGET,
    recent_own_actions: list[Action] | None = None,
    position_history: list[frozenset] | None = None,
) -> Action | None:
    """Return the best action using deadline-aware iterative deepening.

    Tactical shortcuts are kept at root only: immediate wins, then safe
    captures. Otherwise, search progressively deeper until the turn budget is
    used, returning the best fully completed depth.
    """
    deadline = time.perf_counter() + time_budget
    actions = get_all_actions(board, color)

    for action in actions:
        board.apply_action(action)
        try:
            if board.game_over and board.winner_color == color:
                return action
        finally:
            board.undo_action()

    tactical_eat = find_tactical_capture(board, color, actions)
    if tactical_eat is not None:
        return tactical_eat

    search_actions = _filter_root_cascade_actions(board, color, actions)
    if not search_actions:
        return None

    completed_best = search_actions[0]
    completed_score = -math.inf
    max_depth = max(1, depth)

    for current_depth in range(1, max_depth + 1):
        depth_best = None
        depth_score = -math.inf
        try:
            for action in search_actions:
                _check_deadline(deadline)
                root_adjustment = _root_action_adjustment(
                    board, color, action, recent_own_actions or []
                )
                board.apply_action(action)
                try:
                    score = alphabeta(
                        board,
                        current_depth - 1,
                        -math.inf,
                        math.inf,
                        False,
                        color,
                        deadline,
                    )
                    position_visits = 0
                    if position_history:
                        board_hash = frozenset(
                            (coord, cell.color, cell.height)
                            for coord, cell in board._state.items()
                            if not cell.is_empty
                        )
                        position_visits = position_history.count(board_hash)
                finally:
                    board.undo_action()
                score += root_adjustment
                if position_visits == 1:
                    score -= 80.0
                elif position_visits == 2:
                    score -= 160.0
                elif position_visits >= 3:
                    score -= 240.0

                if score > depth_score:
                    depth_score = score
                    depth_best = action
        except _SearchTimeout:
            break

        if depth_best is not None:
            completed_best = depth_best
            completed_score = depth_score

        if abs(completed_score) >= TERMINAL_WIN:
            break

    return completed_best


def alphabeta(
    board: SearchBoard,
    depth: int,
    alpha: float,
    beta: float,
    maximising: bool,
    color: PlayerColor,
    deadline: float | None = None,
) -> float:
    _check_deadline(deadline)

    if board.game_over:
        return _terminal_score(board, color, depth)

    if depth == 0:
        capture_score = evaluate_best_safe_capture_leaf(board, color, deadline)
        if capture_score is not None:
            return capture_score
        return evaluate(board, color)

    actions = get_all_actions(board, board._turn_color)

    if not actions:
        return evaluate(board, color)

    if maximising:
        value = -math.inf
        for action in actions:
            _check_deadline(deadline)
            board.apply_action(action)
            try:
                value = max(
                    value,
                    alphabeta(
                        board, depth - 1, alpha, beta, False, color, deadline
                    ),
                )
            finally:
                board.undo_action()
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value
    else:
        value = math.inf
        for action in actions:
            _check_deadline(deadline)
            board.apply_action(action)
            try:
                value = min(
                    value,
                    alphabeta(
                        board, depth - 1, alpha, beta, True, color, deadline
                    ),
                )
            finally:
                board.undo_action()
            beta = min(beta, value)
            if value <= alpha:
                break
        return value


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise _SearchTimeout()


def _terminal_score(board: SearchBoard, color: PlayerColor, depth_left: int) -> float:
    winner = board.winner_color
    if winner is None:
        return 0.0
    if winner == color:
        return TERMINAL_WIN + depth_left
    return -TERMINAL_WIN - depth_left


def find_tactical_capture(
    board: SearchBoard, color: PlayerColor, actions: list[Action]
) -> Action | None:
    eat_actions = [action for action in actions if isinstance(action, EatAction)]
    safe_eats: list[tuple[int, int, Action]] = []

    for action in eat_actions:
        before_enemy_tokens = _token_count(board, color.opponent)
        before_enemy_stacks = _stack_count(board, color.opponent)
        target_height = _target_height(board, action)

        board.apply_action(action)
        try:
            if board.game_over:
                if board.winner_color == color:
                    return action
                continue

            after_enemy_tokens = _token_count(board, color.opponent)
            after_enemy_stacks = _stack_count(board, color.opponent)
            tokens_removed = before_enemy_tokens - after_enemy_tokens
            stacks_removed = before_enemy_stacks - after_enemy_stacks

            if tokens_removed <= 0 and stacks_removed <= 0:
                continue
            if opponent_has_immediate_win(board, color.opponent):
                continue
            if _is_bad_high_stack_capture(board, action, target_height):
                continue

            safe_eats.append((tokens_removed, stacks_removed, action))
        finally:
            board.undo_action()

    if not safe_eats:
        return None
    return max(safe_eats, key=lambda item: (item[0], item[1]))[2]


def evaluate_best_safe_capture_leaf(
    board: SearchBoard, perspective: PlayerColor, deadline: float | None = None
) -> float | None:
    color = board._turn_color
    scores = []

    for action in get_all_actions(board, color):
        _check_deadline(deadline)
        if not isinstance(action, EatAction):
            continue
        before_enemy_tokens = _token_count(board, color.opponent)
        before_enemy_stacks = _stack_count(board, color.opponent)

        board.apply_action(action)
        try:
            if board.game_over:
                scores.append(_terminal_score(board, perspective, 0))
                continue

            tokens_removed = before_enemy_tokens - _token_count(board, color.opponent)
            stacks_removed = before_enemy_stacks - _stack_count(board, color.opponent)
            if tokens_removed <= 0 and stacks_removed <= 0:
                continue
            if opponent_has_immediate_win(board, color.opponent, deadline):
                continue

            scores.append(evaluate(board, perspective))
        finally:
            board.undo_action()

    if not scores:
        return None
    if color == perspective:
        return max(scores)
    return min(scores)


def opponent_has_immediate_win(
    board: SearchBoard, opponent: PlayerColor, deadline: float | None = None
) -> bool:
    for action in get_all_actions(board, opponent):
        _check_deadline(deadline)
        board.apply_action(action)
        try:
            if board.game_over and board.winner_color == opponent:
                return True
        finally:
            board.undo_action()
    return False


def source_stack_has_legal_eat(
    board: SearchBoard, color: PlayerColor, coord: Coord
) -> bool:
    cell = board._state.get(coord)
    if cell is None or cell.color != color:
        return False

    for direction in CARDINAL_DIRECTIONS:
        try:
            dest = coord + direction
        except ValueError:
            continue

        target = board._state[dest]
        if target.color == color.opponent and cell.height >= target.height:
            return True
    return False


def _root_action_adjustment(
    board: SearchBoard,
    color: PlayerColor,
    action: Action,
    recent_own_actions: list[Action],
) -> float:
    adjustment = 0.0

    if isinstance(action, MoveAction) and source_stack_has_legal_eat(
        board, color, action.coord
    ):
        adjustment -= 50.0

    if isinstance(action, MoveAction) and _reverses_previous_move(
        board, action, recent_own_actions
    ):
        adjustment -= 15.0
        if _recreates_two_ply_cycle(action, recent_own_actions):
            adjustment -= 30.0

    if isinstance(action, CascadeAction) and _is_bad_non_capture_cascade(
        board, color, action
    ):
        adjustment -= 150.0

    return adjustment


def _filter_root_cascade_actions(
    board: SearchBoard, color: PlayerColor, actions: list[Action]
) -> list[Action]:
    filtered = [
        action for action in actions
        if not isinstance(action, CascadeAction)
        or not _is_bad_non_capture_cascade(board, color, action)
    ]
    return filtered or actions


def _is_bad_non_capture_cascade(
    board: SearchBoard, color: PlayerColor, action: CascadeAction
) -> bool:
    if _cascade_removes_enemy(board, color, action):
        return False
    if _cascade_targets_height_one_enemy(board, color, action):
        return False
    if _cascade_merges_friendly_stack(board, color, action):
        return False
    return True


def _cascade_removes_enemy(
    board: SearchBoard, color: PlayerColor, action: CascadeAction
) -> bool:
    before_enemy_tokens = _token_count(board, color.opponent)
    before_enemy_stacks = _stack_count(board, color.opponent)

    board.apply_action(action)
    try:
        after_enemy_tokens = _token_count(board, color.opponent)
        after_enemy_stacks = _stack_count(board, color.opponent)
        return (
            after_enemy_tokens < before_enemy_tokens
            or after_enemy_stacks < before_enemy_stacks
        )
    finally:
        board.undo_action()


def _cascade_targets_height_one_enemy(
    board: SearchBoard, color: PlayerColor, action: CascadeAction
) -> bool:
    source = board._state.get(action.coord)
    if source is None or source.color != color:
        return False

    for steps in range(1, source.height + 1):
        target = _destination(action.coord, action.direction, steps)
        if target is None:
            continue

        cell = board._state[target]
        if cell.color == color.opponent and cell.height == 1:
            return True

    return False


def _cascade_merges_friendly_stack(
    board: SearchBoard, color: PlayerColor, action: CascadeAction
) -> bool:
    source = board._state.get(action.coord)
    if source is None or source.color != color or source.height < 2:
        return False

    dest = _destination(action.coord, action.direction)
    if dest is None:
        return False

    target = board._state[dest]
    if target.color != color:
        return False

    exposed_before = _height_one_exposure_count(board, color)
    board.apply_action(action)
    try:
        exposed_after = _height_one_exposure_count(board, color)
    finally:
        board.undo_action()

    return exposed_after <= exposed_before + 1


def _height_one_exposure_count(board: SearchBoard, color: PlayerColor) -> int:
    count = 0

    for coord, cell in board._state.items():
        if cell.color != color:
            continue

        for direction in CARDINAL_DIRECTIONS:
            dest = _destination(coord, direction)
            if dest is None:
                continue

            target = board._state[dest]
            if target.color == color.opponent and target.height == 1:
                count += 1
                break

    return count


def _reverses_previous_move(
    board: SearchBoard, action: MoveAction, recent_own_actions: list[Action]
) -> bool:
    if not recent_own_actions:
        return False

    previous = recent_own_actions[-1]
    if not isinstance(previous, MoveAction):
        return False

    try:
        previous_dest = previous.coord + previous.direction
    except ValueError:
        return False

    if action.coord != previous_dest or action.direction != -previous.direction:
        return False

    cell = board._state.get(action.coord)
    return cell is not None and not cell.is_empty


def _recreates_two_ply_cycle(
    action: MoveAction, recent_own_actions: list[Action]
) -> bool:
    if len(recent_own_actions) < 2:
        return False

    first = recent_own_actions[-2]
    second = recent_own_actions[-1]
    if not isinstance(first, MoveAction) or not isinstance(second, MoveAction):
        return False

    try:
        first_dest = first.coord + first.direction
    except ValueError:
        return False

    return action.coord == first_dest and action.direction == -second.direction


def _is_bad_high_stack_capture(
    board_after: SearchBoard, action: EatAction, target_height: int
) -> bool:
    dest = _destination(action.coord, action.direction)
    if dest is None:
        return False

    cell = board_after._state[dest]
    if cell.is_empty or cell.height < 4:
        return False

    if _is_corner(dest):
        return target_height < 2
    if _is_edge(dest):
        return target_height < 2
    return False


def _target_height(board: SearchBoard, action: EatAction) -> int:
    dest = _destination(action.coord, action.direction)
    if dest is None:
        return 0
    return board._state[dest].height


def _destination(coord: Coord, direction: Direction, steps: int = 1) -> Coord | None:
    try:
        return Coord(coord.r + direction.r * steps, coord.c + direction.c * steps)
    except ValueError:
        return None


def _token_count(board: SearchBoard, color: PlayerColor) -> int:
    return sum(cell.height for cell in board._state.values() if cell.color == color)


def _stack_count(board: SearchBoard, color: PlayerColor) -> int:
    return sum(1 for cell in board._state.values() if cell.color == color)


def _is_edge(coord: Coord) -> bool:
    return (
        coord.r == 0
        or coord.r == BOARD_SIZE - 1
        or coord.c == 0
        or coord.c == BOARD_SIZE - 1
    )


def _is_corner(coord: Coord) -> bool:
    edge_values = {0, BOARD_SIZE - 1}
    return coord.r in edge_values and coord.c in edge_values


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Cascade game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._board = Board()
        self._search_board = SearchBoard()
        self._color = color
        self._turn_count = 0
        self._recent_own_actions = []
        self._position_history: list[frozenset] = []
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """
        if self._turn_count < 4:
            return self._best_placement_action()

        action = find_best_action(
            self._search_board,
            self._color,
            recent_own_actions=self._recent_own_actions,
            position_history=self._position_history,
        )
        if action is not None:
            return action

        return MoveAction(Coord(0, 0), Direction.Down)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        if color == self._color:
            self._turn_count += 1
            self._recent_own_actions.append(action)
            self._recent_own_actions = self._recent_own_actions[-4:]

        self._board.apply_action(color, action)
        self._search_board.apply_action(action)
        board_hash = frozenset(
            (coord, cell.color, cell.height)
            for coord, cell in self._search_board._state.items()
            if not cell.is_empty
        )
        self._position_history.append(board_hash)
        self._position_history = self._position_history[-10:]

        match action:
            case PlaceAction(coord):
                print(f"Testing: {color} played PLACE action at {coord}")
            case MoveAction(coord, direction):
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")
            case EatAction(coord, direction):
                print(f"Testing: {color} played EAT action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")
            case CascadeAction(coord, direction):
                print(f"Testing: {color} played CASCADE action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")
            case _:
                raise ValueError(f"Unknown action type: {action}")

    def get_eval(self, board: Board) -> float:
        return evaluate(board, self._color)

    def _best_placement_action(self) -> PlaceAction:
        candidates = self._valid_place_coords()
        if not candidates:
            return PlaceAction(Coord(0, 0))

        best_coord = max(
            candidates,
            key=lambda coord: evaluate_placement(
                self._search_board, self._color, coord
            ),
        )
        return PlaceAction(best_coord)

    def _valid_place_coords(self) -> list[Coord]:
        occupied = {
            coord for coord, cell in self._search_board._state.items()
            if cell.is_stack
        }
        enemy_coords = [
            coord for coord, cell in self._search_board._state.items()
            if cell.color == self._color.opponent
        ]

        candidates = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                coord = Coord(r, c)
                if coord in occupied:
                    continue
                if enemy_coords and any(
                    _is_adjacent(coord, enemy) for enemy in enemy_coords
                ):
                    continue
                candidates.append(coord)

        return candidates


def _is_adjacent(coord_a: Coord, coord_b: Coord) -> bool:
    return abs(coord_a.r - coord_b.r) + abs(coord_a.c - coord_b.c) == 1
