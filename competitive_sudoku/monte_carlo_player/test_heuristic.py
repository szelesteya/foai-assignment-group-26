import numpy as np
import pytest

from pathlib import Path
import sys

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from node import SudokuNode  # noqa: E402


def test_evaluate_score_difference():
    # 4x4 board
    board = np.zeros((4, 4), dtype=np.int32)
    empty_moves = np.empty((0, 3), dtype=np.int32)

    # Node where player 1 has 5 points and player 2 has 3 points
    # and it's currently player 2's turn (meaning player 1 just moved)
    node = SudokuNode(
        game_board=board,
        current_player=2,
        played_moves_player1=empty_moves,
        played_moves_player2=empty_moves,
        player_1_score=5,
        player_2_score=3,
        proposed_taboo_moves_player1=empty_moves,
        proposed_taboo_moves_player2=empty_moves,
        box_height=2,
        box_width=2,
    )

    # Evaluate from perspective of player 1 (who just moved)
    # real_score_diff = 5 - 3 = 2
    # mobility_score depends on reach
    # Reach P1: 4 (row 0), Reach P2: 4 (row 3)
    # mobility_score = 4 - 3*4 = -8
    # score = 2 * 20000 + (-8 * 200) = 40000 - 1600 = 38400
    score = node.evaluate()

    assert score == 38400.0


def test_evaluate_mobility():
    board = np.zeros((4, 4), dtype=np.int32)
    # Mark (0,0) as played on the board to exclude it from reach
    board[0, 0] = 1
    empty_moves = np.empty((0, 3), dtype=np.int32)

    # Player 1 has played at (0,0), giving them neighbors (0,1), (1,0), (1,1)
    # Plus first row (0,1), (0,2), (0,3).
    # Since (0,0) is occupied, P1 reach is: (0,1), (0,2), (0,3), (1,0), (1,1) -> 5 cells
    played_p1 = np.array([[0, 0, 1]], dtype=np.int32)

    # Player 2 has no moves yet. Reach P2: (3,0), (3,1), (3,2), (3,3) -> 4 cells

    node = SudokuNode(
        game_board=board,
        current_player=2,
        played_moves_player1=played_p1,
        played_moves_player2=empty_moves,
        player_1_score=0,
        player_2_score=0,
        proposed_taboo_moves_player1=empty_moves,
        proposed_taboo_moves_player2=empty_moves,
        box_height=2,
        box_width=2,
    )

    # mobility_score = 5 - (3.0 * 4) = 5 - 12 = -7
    # evaluate() should include -7 * 200 = -1400
    score = node.evaluate()
    assert score == -1400.0


def test_evaluate_denial_bonus():
    board = np.zeros((4, 4), dtype=np.int32)
    empty_moves = np.empty((0, 3), dtype=np.int32)

    node = SudokuNode(
        game_board=board,
        current_player=2,
        played_moves_player1=empty_moves,
        played_moves_player2=empty_moves,
        player_1_score=0,
        player_2_score=0,
        proposed_taboo_moves_player1=empty_moves,
        proposed_taboo_moves_player2=empty_moves,
        box_height=2,
        box_width=2,
    )

    # Current reach P2 is 4 cells (last row)
    # If initial_opp_reach was 6 cells, denial_bonus = 6 - 4 = 2
    # 2 * 500 = 1000

    # Reach P1 is 4 cells (first row)
    # Reach P2 is 4 cells (last row)
    # mobility_score = 4 - (3.0 * 4) = -8
    # -8 * 200 = -1600

    initial_opp_reach = np.array([[3, 0], [3, 1], [3, 2], [3, 3], [2, 0], [2, 1]], dtype=np.int32)
    score = node.evaluate(initial_opp_reach=initial_opp_reach)

    # Expected: -1600 + 1000 = -600
    assert score == -600.0


def test_find_single_holes():
    # Board with row 0 almost complete (needs 4 at (0,3))
    board = np.array([[1, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)

    node = SudokuNode(
        game_board=board,
        current_player=1,
        played_moves_player1=np.empty((0, 3), dtype=np.int32),
        played_moves_player2=np.empty((0, 3), dtype=np.int32),
        player_1_score=0,
        player_2_score=0,
        proposed_taboo_moves_player1=np.empty((0, 3), dtype=np.int32),
        proposed_taboo_moves_player2=np.empty((0, 3), dtype=np.int32),
        box_height=2,
        box_width=2,
    )

    holes = node._find_single_holes()
    assert len(holes) == 1
    assert tuple(holes[0]) == (0, 3)


if __name__ == "__main__":
    pytest.main([__file__])
