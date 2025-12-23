#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from competitive_sudoku.sudoku import GameState, Move, TabooMove
import competitive_sudoku.sudokuai
import numpy as np
import logging

from .mcts import MCTS
from .node import SudokuNode

ROLLOUTS_PER_MOVE = 1000

logger = logging.getLogger(__name__)


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration using Monte Carlo Tree Search.
    """

    def __init__(self):
        super().__init__()

    def _propose_mcts_move(
        self,
        mcts: MCTS,
        root_node: SudokuNode,
        game_state: GameState,
        taboo_arr: np.ndarray,
    ) -> None:
        """Helper to choose and propose the best move found so far by MCTS."""
        try:
            if root_node not in mcts.children or not mcts.children[root_node]:
                return

            best_child = mcts.choose(root_node)
            is_p1 = game_state.current_player == 1

            # Identify the move
            old_moves = root_node.played_moves_player1 if is_p1 else root_node.played_moves_player2
            new_moves = best_child.played_moves_player1 if is_p1 else best_child.played_moves_player2

            move_to_propose = None
            if len(new_moves) > len(old_moves):
                m = new_moves[-1]
                move_to_propose = Move((int(m[0]), int(m[1])), int(m[2]))
            else:
                # Check if it was a taboo move
                old_taboo = taboo_arr
                new_taboo = (
                    best_child.proposed_taboo_moves_player1 if is_p1 else best_child.proposed_taboo_moves_player2
                )
                if len(new_taboo) > len(old_taboo):
                    m = new_taboo[-1]
                    move_to_propose = Move((int(m[0]), int(m[1])), int(m[2]))

            if move_to_propose:
                self.propose_move(move_to_propose)
        except Exception as e:
            logger.error(f"Error in _propose_mcts_move: {e}")

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Computes the best move using MCTS and proposes it.
        """
        try:
            # 1. Reconstruct root node
            n = game_state.board.N
            game_board = np.array(game_state.board.squares, dtype=np.int32).reshape((n, n))

            occupied1 = set(game_state.occupied_squares1) if game_state.occupied_squares1 else set()
            occupied2 = set(game_state.occupied_squares2) if game_state.occupied_squares2 else set()

            pm1 = []
            pm2 = []
            for m in game_state.moves:
                if isinstance(m, TabooMove):
                    continue
                if m.square in occupied1:
                    pm1.append([m.square[0], m.square[1], m.value])
                elif m.square in occupied2:
                    pm2.append([m.square[0], m.square[1], m.value])

            played_moves_p1 = np.array(pm1, dtype=np.int32) if pm1 else np.empty((0, 3), dtype=np.int32)
            played_moves_p2 = np.array(pm2, dtype=np.int32) if pm2 else np.empty((0, 3), dtype=np.int32)

            # All taboo moves are known to both for simplicity as we don't know who proposed them
            taboo_moves_all = []
            for tm in game_state.taboo_moves:
                taboo_moves_all.append([tm.square[0], tm.square[1], tm.value])

            taboo_arr = (
                np.array(taboo_moves_all, dtype=np.int32) if taboo_moves_all else np.empty((0, 3), dtype=np.int32)
            )

            root_node = SudokuNode(
                game_board=game_board,
                current_player=game_state.current_player,
                played_moves_player1=played_moves_p1,
                played_moves_player2=played_moves_p2,
                player_1_score=game_state.scores[0],
                player_2_score=game_state.scores[1],
                proposed_taboo_moves_player1=taboo_arr,
                proposed_taboo_moves_player2=taboo_arr,
                box_height=game_state.board.m,
                box_width=game_state.board.n,
            )

            # 2. Run MCTS
            mcts = MCTS()

            # Initial rollouts to ensure we have at least one move proposed
            # before potentially being killed by a short timeout.
            for _ in range(10):
                mcts.do_rollout(root_node)

            self._propose_mcts_move(mcts, root_node, game_state, taboo_arr)

            # 3. Continue running rollouts until process is terminated
            # We use a large number and rely on simulation timeout.
            for i in range(ROLLOUTS_PER_MOVE):
                mcts.do_rollout(root_node)
                # Periodically update the proposed move
                if (i + 1) % 50 == 0:
                    self._propose_mcts_move(mcts, root_node, game_state, taboo_arr)

        except Exception as e:
            print(e)
            logger.error("Error in compute_best_move", exc_info=True)
