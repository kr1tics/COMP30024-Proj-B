from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.program import find_best_action, get_all_actions
from referee.game import (
    Board,
    CascadeAction,
    CellState,
    Coord,
    Direction,
    EatAction,
    MoveAction,
    PlaceAction,
    PlayerColor,
)


ACTION_RE = re.compile(r"(PLACE|MOVE|EAT|CASCADE)\((\d)-(\d)(?:, \[(.)\])?\)")
DIRECTIONS = {
    "↑": Direction.Up,
    "↓": Direction.Down,
    "←": Direction.Left,
    "→": Direction.Right,
}


def parse_action(text: str):
    match = ACTION_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"Could not parse action: {ascii(text)}")

    kind, row, col, direction_text = match.groups()
    coord = Coord(int(row), int(col))

    if kind == "PLACE":
        return PlaceAction(coord)

    direction = DIRECTIONS[direction_text]
    if kind == "MOVE":
        return MoveAction(coord, direction)
    if kind == "EAT":
        return EatAction(coord, direction)
    if kind == "CASCADE":
        return CascadeAction(coord, direction)

    raise ValueError(f"Unknown action kind: {kind}")


def board_after_turn(log_name: str, turn: int) -> Board:
    board = Board()
    path = ROOT / log_name

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 5 or parts[2] != "turn_end":
            continue
        if int(parts[3]) > turn:
            break
        board.apply_action(parse_action(parts[4]))

    return board


def assert_game12_capture() -> None:
    board = board_after_turn("game12.log", 54)
    expected = EatAction(Coord(2, 4), Direction.Right)

    assert expected in get_all_actions(board, PlayerColor.RED)
    assert find_best_action(board, PlayerColor.RED) == expected


def assert_repeated_cascade_pattern(log_name: str) -> None:
    board = board_after_turn(log_name, 54)
    expected = EatAction(Coord(2, 4), Direction.Right)
    bad_runaway = MoveAction(Coord(2, 4), Direction.Up)

    actions = get_all_actions(board, PlayerColor.RED)
    assert expected in actions
    assert find_best_action(board, PlayerColor.RED) == expected
    assert find_best_action(board, PlayerColor.RED) != bad_runaway


def assert_minimal_pattern() -> None:
    board = Board(
        {
            Coord(0, 5): CellState(PlayerColor.BLUE, 1),
            Coord(1, 5): CellState(PlayerColor.BLUE, 1),
            Coord(2, 4): CellState(PlayerColor.RED, 5),
            Coord(2, 5): CellState(PlayerColor.BLUE, 1),
        },
        initial_player=PlayerColor.RED,
    )
    board.set_placement_count(8)

    expected = EatAction(Coord(2, 4), Direction.Right)
    bad_runaway = MoveAction(Coord(2, 4), Direction.Up)

    assert expected in get_all_actions(board, PlayerColor.RED)
    assert find_best_action(board, PlayerColor.RED) == expected
    assert find_best_action(board, PlayerColor.RED) != bad_runaway


if __name__ == "__main__":
    assert_game12_capture()
    assert_repeated_cascade_pattern("game11.log")
    assert_repeated_cascade_pattern("game13.log")
    assert_minimal_pattern()
    print("capture regression checks passed")
