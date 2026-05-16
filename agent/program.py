# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

import math

from agent.board import Board
from agent.evaluate import TERMINAL_WIN, evaluate, evaluate_placement
from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game import Board as SearchBoard


BOARD_SIZE = 8


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
    depth: int = 3,
    recent_own_actions: list[Action] | None = None,
) -> Action | None:
    """Return the best action using the original stable action order.

    Tactical shortcuts are kept at root only: immediate wins, then safe
    captures. Otherwise, keep the old search behaviour so the agent does not
    over-prioritise risky cascades or speculative moves.
    """
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

    best_action = None
    best_score = -math.inf

    for action in actions:
        root_adjustment = _root_action_adjustment(
            board, color, action, recent_own_actions or []
        )
        board.apply_action(action)
        try:
            score = alphabeta(board, depth - 1, -math.inf, math.inf, False, color)
        finally:
            board.undo_action()
        score += root_adjustment

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def alphabeta(
    board: SearchBoard,
    depth: int,
    alpha: float,
    beta: float,
    maximising: bool,
    color: PlayerColor,
) -> float:
    if board.game_over:
        return _terminal_score(board, color, depth)

    if depth == 0:
        capture_score = evaluate_best_safe_capture_leaf(board, color)
        if capture_score is not None:
            return capture_score
        return evaluate(board, color)

    actions = get_all_actions(board, board._turn_color)

    if not actions:
        return evaluate(board, color)

    if maximising:
        value = -math.inf
        for action in actions:
            board.apply_action(action)
            value = max(value, alphabeta(board, depth - 1, alpha, beta, False, color))
            board.undo_action()
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value
    else:
        value = math.inf
        for action in actions:
            board.apply_action(action)
            value = min(value, alphabeta(board, depth - 1, alpha, beta, True, color))
            board.undo_action()
            beta = min(beta, value)
            if value <= alpha:
                break
        return value


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
    board: SearchBoard, perspective: PlayerColor
) -> float | None:
    color = board._turn_color
    scores = []

    for action in get_all_actions(board, color):
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
            if opponent_has_immediate_win(board, color.opponent):
                continue

            scores.append(evaluate(board, perspective))
        finally:
            board.undo_action()

    if not scores:
        return None
    if color == perspective:
        return max(scores)
    return min(scores)


def opponent_has_immediate_win(board: SearchBoard, opponent: PlayerColor) -> bool:
    for action in get_all_actions(board, opponent):
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

    return adjustment


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
    return cell is not None and cell.height >= 4


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


def _destination(coord: Coord, direction: Direction) -> Coord | None:
    try:
        return coord + direction
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
            self._search_board, self._color, recent_own_actions=self._recent_own_actions
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
