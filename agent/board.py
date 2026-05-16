from referee.game import PlaceAction, PlayerColor


class Board:
    def __init__(self):
        self._stacks = {
            PlayerColor.RED: {},
            PlayerColor.BLUE: {},
        }

    def get_stacks(self, color):
        return list(self._stacks[color].items())

    def total_tokens(self, color):
        return sum(self._stacks[color].values())

    def is_terminal(self):
        return False, None

    def copy(self):
        board = Board()
        board._stacks = {
            PlayerColor.RED: dict(self._stacks[PlayerColor.RED]),
            PlayerColor.BLUE: dict(self._stacks[PlayerColor.BLUE]),
        }
        return board

    def apply_action(self, color, action):
        if isinstance(action, PlaceAction):
            self._stacks[color][action.coord] = 3
