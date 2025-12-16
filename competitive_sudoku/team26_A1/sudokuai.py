#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from competitive_sudoku.sudoku import GameState, Move, TabooMove, SudokuBoard
import competitive_sudoku.sudokuai
from team26_A1.evaluate import evaluate, is_valid_board
import copy
import math


MAX_DEPTH = 2


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def _minimax(
        self,
        game_state: GameState,
        depth: int,
        maximizing_player: bool,
        alpha: int,
        beta: int,
    ) -> int:
        N = game_state.board.N

        # Get valid squares for the current player
        # Create a temporary game state to get player_squares for the right player
        temp_state = copy.deepcopy(game_state)
        temp_state.current_player = 1 if maximizing_player else 2
        squares = temp_state.player_squares()

        if depth == MAX_DEPTH or squares is None or len(squares) == 0:
            return evaluate(game_state, maximizing_player, depth)

        if maximizing_player:
            best_value = alpha
            found_valid_move = False
            for square in squares:
                # Only consider empty squares
                if game_state.board.get(square) != SudokuBoard.empty:
                    continue

                for num in range(1, N + 1):
                    move = Move(square, num)

                    # Skip taboo moves
                    if TabooMove(square, num) in game_state.taboo_moves:
                        continue

                    # Create new game state to check if move would be valid
                    new_game_state = _game_copy_w_move(
                        game_state=game_state, move=move, maximizing_player=True
                    )

                    # Skip moves that would create invalid boards (duplicates)
                    # if not is_valid_board(new_game_state.board):
                    #     continue

                    found_valid_move = True
                    score = self._minimax(
                        new_game_state,
                        depth + 1,
                        False,
                        alpha,
                        beta,
                    )
                    best_value = max(best_value, score)
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        # print(
                        #     f"Pruning! alpha={alpha} >= beta={beta}, score={score}, move={move}"
                        # )
                        return alpha
            # If no valid moves were found, return evaluation
            return (
                evaluate(game_state, maximizing_player, depth)
                if not found_valid_move
                else best_value
            )
        else:
            best_value = beta
            found_valid_move = False
            for square in squares:
                # Only consider empty squares
                if game_state.board.get(square) != SudokuBoard.empty:
                    continue

                for num in range(1, N + 1):
                    move = Move(square, num)

                    # Skip taboo moves
                    if TabooMove(square, num) in game_state.taboo_moves:
                        continue

                    # Create new game state to check if move would be valid
                    new_game_state = _game_copy_w_move(
                        game_state=game_state, move=move, maximizing_player=False
                    )

                    # Skip moves that would create invalid boards (duplicates)
                    # if not is_valid_board(new_game_state.board):
                    #     continue

                    found_valid_move = True
                    score = self._minimax(
                        new_game_state,
                        depth + 1,
                        True,
                        alpha,
                        beta,
                    )
                    best_value = min(best_value, score)
                    beta = min(beta, score)
                    if alpha >= beta:
                        # print(
                        #     f"Pruning! alpha={alpha} >= beta={beta}, score={score}, move={move}"
                        # )
                        return beta
            # If no valid moves were found, return evaluation
            return (
                evaluate(game_state, maximizing_player, depth)
                if not found_valid_move
                else best_value
            )

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        alpha = -999999999  # Start with -infinity for maximizing
        beta = 999999999  # Start with +infinity for minimizing
        best_move = Move((0, 0), 1)
        best_score = -999999999  # Initialize best_score

        squares = game_state.player_squares()
        if squares is None or len(squares) == 0:
            # No valid moves, propose a default move
            self.propose_move(best_move)
            return

        for square in squares:
            # Only consider empty squares
            if game_state.board.get(square) != SudokuBoard.empty:
                continue

            for value in range(1, N + 1):
                move = Move(square, value)

                # Skip taboo moves
                if TabooMove(square, value) in game_state.taboo_moves:
                    continue

                # Create new game state to check if move would be valid
                new_game_state = _game_copy_w_move(
                    game_state=game_state, move=move, maximizing_player=True
                )

                # Skip moves that would create invalid boards (duplicates)
                # if not is_valid_board(new_game_state.board):
                #     continue

                # After we make our move, opponent plays next (minimizing from our perspective)
                score = self._minimax(
                    new_game_state,
                    0,
                    False,  # Opponent plays next (minimizing)
                    alpha,
                    beta,
                )
                # print(f"I dont get it {score}, {move}")
                if score > best_score:
                    best_move = move
                    best_score = score
                # Update alpha at root level for alpha-beta pruning
                alpha = max(alpha, score)
            # print(f"truly notttt {best_score}, {best_move}")

        # print(f"Best move is..... : {best_move}")
        # print(f"Return from compute best move")

        self.propose_move(best_move)


def _game_copy_w_move(
    game_state: GameState, move: Move, maximizing_player: bool = None
) -> GameState:
    game_copy = copy.deepcopy(game_state)

    # Only apply move if square is empty
    if game_copy.board.get(move.square) != SudokuBoard.empty:
        return game_copy

    # Apply the move to the board
    game_copy.board.put(move.square, move.value)

    # Update moves list
    game_copy.moves.append(move)

    # Switch current player
    if maximizing_player is not None:
        # If we know which player made the move, set current_player to the other player
        game_copy.current_player = 2 if maximizing_player else 1
    else:
        # Otherwise, just switch players
        game_copy.current_player = 3 - game_copy.current_player

    return game_copy
