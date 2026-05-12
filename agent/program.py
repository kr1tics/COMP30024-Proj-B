# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from agent.board import Board
from agent.evaluate import evaluate, evaluate_placement, rank_actions
from referee.game import PlayerColor, Coord, Direction, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction


BOARD_SIZE = 8


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
        # During placement phase (first 8 turns total, 4 per player)
        if self._turn_count < 4:
            return self._best_placement_action()

        # During play phase
        legal_actions = self._legal_actions()
        if legal_actions:
            return rank_actions(self._board, self._color, legal_actions)[0][1]

        return MoveAction(Coord(0, 0), Direction.Down)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        if color == self._color:
            self._turn_count += 1

        self._board.apply_action(color, action)
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
            key=lambda coord: evaluate_placement(self._board, self._color, coord),
        )
        return PlaceAction(best_coord)

    def _valid_place_coords(self) -> list[Coord]:
        occupied = {
            coord
            for player_color in PlayerColor
            for coord, _ in self._board.get_stacks(player_color)
        }
        enemy_coords = [
            coord for coord, _ in self._board.get_stacks(self._color.opponent)
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

    def _legal_actions(self) -> list[Action]:
        for method_name in ("get_legal_actions", "legal_actions", "actions"):
            method = getattr(self._board, method_name, None)
            if method is None:
                continue

            try:
                return list(method(self._color))
            except TypeError:
                return list(method())

        return []


def _is_adjacent(coord_a: Coord, coord_b: Coord) -> bool:
    return abs(coord_a.r - coord_b.r) + abs(coord_a.c - coord_b.c) == 1
