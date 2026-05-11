class Board:
    def get_stacks(self, color):
        return []

    def total_tokens(self, color):
        return 0

    def is_terminal(self):
        return False, None

    def copy(self):
        return Board()

    def apply_action(self, color, action):
        pass
