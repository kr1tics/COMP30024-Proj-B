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
    board: SearchBoard, color: PlayerColor, depth: int = 3
) -> Action | None:
    """Return the best action using the original stable action order.

    Only one tactical shortcut is added: if a legal action wins immediately,
    take it. Otherwise, keep the old search behaviour so the agent does not
    over-prioritise risky cascades or speculative moves.
    """
    for action in get_all_actions(board, color):
        board.apply_action(action)
        try:
            if board.game_over and board.winner_color == color:
                return action
        finally:
            board.undo_action()

    best_action = None
    best_score = -math.inf

    for action in get_all_actions(board, color):
        board.apply_action(action)
        try:
            score = alphabeta(board, depth - 1, -math.inf, math.inf, False, color)
        finally:
            board.undo_action()

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

        action = find_best_action(self._search_board, self._color)
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
