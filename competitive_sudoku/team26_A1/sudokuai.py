#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from competitive_sudoku.sudoku import GameState, Move
import competitive_sudoku.sudokuai
from competitive_sudoku.team26_A1.evaluate import evaluate


MAX_DEPTH = 4


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def _minimax(self, game_state: GameState, depth: int, maximizing_player: bool) -> int:
        N = game_state.board.N
        squares = game_state.allowed_squares1 if maximizing_player else game_state.allowed_squares2
        if depth == MAX_DEPTH or squares is None or len(squares) == 0:
            return evaluate(game_state, maximizing_player, depth)

        for num in range(N):
            if maximizing_player:
                max_eval = float("-inf")
                for square in squares:
                    move = Move(square, num)
                    board_w_move = game_state.board.put(move.square, move.value)
                    eval = self._minimax(GameState(board=board_w_move), depth + 1, False)
                    max_eval = max(max_eval, eval)
                return max_eval
            else:
                min_eval = float("inf")
                for square in squares:
                    move = Move(square, num)
                    board_w_move = game_state.board.put(move.square, move.value)
                    eval = self._minimax(GameState(board=board_w_move), depth + 1, True)
                    min_eval = min(min_eval, eval)
                return min_eval

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        max_eval = float("-inf")

        for move in game_state.allowed_squares1():
            for value in range(1, N + 1):
                move = Move(move, value)
                board_w_move = game_state.board.put(move.square, move.value)
                score = self._minimax(GameState(board=board_w_move), 0, False)
                if score > max_eval:
                    max_eval = score
                    best_move = move

        self.propose_move(best_move)
