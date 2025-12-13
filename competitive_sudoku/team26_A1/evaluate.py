from competitive_sudoku.sudoku import GameState, is_valid_board


def evaluate(game_state: GameState, is_maximizing_player: bool, depth: int) -> int:
    board = game_state.board

    if board is None:
        raise ValueError("Board is None")

    print("--------------------------------")
    print(game_state.board.squares)
    print("--------------------------------")

    if not is_valid_board(game_state.board):
        inf = float("-inf") if is_maximizing_player else float("inf")
        return inf * depth
