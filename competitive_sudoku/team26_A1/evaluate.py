from math import sqrt
from typing import List
from competitive_sudoku.sudoku import (
    GameState,
    Move,
    SudokuBoard,
    TabooMove,
)

SCORING = {0: 0, 1: 1, 2: 3, 3: 7}


def evaluate(game_state: GameState, is_maximizing_player: bool, depth: int) -> int:
    board = game_state.board

    if board is None:
        raise ValueError("Board is None")

    if not is_valid_board(game_state.board):
        return -1000 * depth

    return 100


def board_score(game_state: GameState) -> int:
    n = game_state.board.N
    m = game_state.board.m
    last_move = game_state.moves[-1]
    if isinstance(last_move, TabooMove):
        return 0
    last_mover = 1 if last_move in game_state.occupied_squares1 else 2
    # Check for regions
    row = last_move.square[0]
    column = last_move.square[1]
    nums_in_row = 0
    nums_in_column = 0
    nums_in_block = 0
    sqrt_N = sqrt(n)
    sqrt_M = sqrt(m)
    block_start_row = (row // sqrt_N) * sqrt_N
    block_end_row = block_start_row + sqrt_N
    block_start_column = (column // sqrt_M) * sqrt_M
    block_end_column = block_start_column + sqrt_M
    for move in game_state.moves:
        move_row = move.square[0]
        move_column = move.square[1]
        if move_row == row:
            nums_in_row += 1
        if move_column == column:
            nums_in_column += 1
        if (
            move_row > block_start_row
            and move_row < block_end_row
            and move_column > block_start_column
            and move_column < block_end_column
        ):
            nums_in_block += 1

    regions_finished = sum(
        1
        for x in [
            nums_in_row == n,
            nums_in_column == m,
            nums_in_block == (sqrt_N * sqrt_M),
        ]
        if x
    )
    return SCORING[regions_finished]


def is_valid_board(board: SudokuBoard) -> bool:
    """
    Check whether the given `board` is currently valid:
    - all non-empty values are within the range [1, N]
    - no duplicate non-empty values appear in any row, column or region

    Returns True if valid, False otherwise.
    """
    N = board.N

    def _has_duplicate(values: List[int]) -> bool:
        seen = set()
        for v in values:
            if v == SudokuBoard.empty:
                continue
            if v < 1 or v > N:
                return True
            if v in seen:
                return True
            seen.add(v)
        return False

    # check rows
    for i in range(N):
        row_vals = [abs(board.get((i, j))) for j in range(N)]
        if _has_duplicate(row_vals):
            return False

    # check columns
    for j in range(N):
        col_vals = [board.get((i, j)) for i in range(N)]
        if _has_duplicate(col_vals):
            return False

    # check regions
    m = board.m
    n = board.n
    for br in range(0, N, m):
        for bc in range(0, N, n):
            block_vals = []
            for i in range(br, br + m):
                for j in range(bc, bc + n):
                    block_vals.append(board.get((i, j)))
            if _has_duplicate(block_vals):
                return False

    return True
