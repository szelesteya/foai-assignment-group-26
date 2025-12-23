"""Unit tests for SudokuNode._get_allowed_moves function."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from node import SudokuNode  # noqa: E402
from solver import solve  # noqa: E402


class TestGetAllowedMoves:
    """Test suite for the _get_allowed_moves method."""

    @pytest.fixture
    def empty_taboo(self) -> np.ndarray:
        """Return empty taboo moves array."""
        return np.empty((0, 3), dtype=np.int32)

    @pytest.fixture
    def empty_played(self) -> np.ndarray:
        """Return empty played moves array."""
        return np.empty((0, 3), dtype=np.int32)

    @staticmethod
    def _create_board(
        size: int,
        played_p1: np.ndarray,
        played_p2: np.ndarray,
        initial_values: list[tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """Create a board with played moves and optional initial values.

        Args:
            size: Board size (NxN).
            played_p1: Player 1's played moves [[row, col, value], ...].
            played_p2: Player 2's played moves [[row, col, value], ...].
            initial_values: Optional list of (row, col, value) for pre-filled cells.

        Returns:
            Board with all moves placed.
        """
        board = np.zeros((size, size), dtype=np.int32)

        # Place initial values (e.g., puzzle starting state)
        if initial_values:
            for row, col, val in initial_values:
                board[row, col] = val

        # Place player 1's moves
        for move in played_p1:
            board[move[0], move[1]] = move[2]

        # Place player 2's moves
        for move in played_p2:
            board[move[0], move[1]] = move[2]

        return board

    def _create_node(
        self,
        game_board: np.ndarray,
        current_player: int,
        played_p1: np.ndarray,
        played_p2: np.ndarray,
        taboo_p1: np.ndarray,
        taboo_p2: np.ndarray,
        box_height: int = 2,
        box_width: int = 2,
        player_1_score: int = 0,
        player_2_score: int = 0,
    ) -> SudokuNode:
        """Helper to create a SudokuNode with given parameters."""
        return SudokuNode(
            game_board=game_board,
            current_player=current_player,
            played_moves_player1=played_p1,
            played_moves_player2=played_p2,
            player_1_score=player_1_score,
            player_2_score=player_2_score,
            proposed_taboo_moves_player1=taboo_p1,
            proposed_taboo_moves_player2=taboo_p2,
            box_height=box_height,
            box_width=box_width,
        )

    @staticmethod
    def _combine_moves(valid: np.ndarray, taboo: np.ndarray) -> np.ndarray:
        """Combine valid and taboo moves into a single array."""
        if len(valid) == 0 and len(taboo) == 0:
            return np.empty((0, 3), dtype=np.int32)
        if len(valid) == 0:
            return taboo
        if len(taboo) == 0:
            return valid
        return np.vstack([valid, taboo])

    def test_empty_played_moves_uses_default_row_player_1(
        self, empty_taboo: np.ndarray, empty_played: np.ndarray
    ) -> None:
        """When Player 1 has no played moves, should return moves from first row (row 0)."""
        board = self._create_board(4, empty_played, empty_played)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=empty_played,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Player 1 should have access to all cells in row 0
        expected_cells = {(0, 0), (0, 1), (0, 2), (0, 3)}
        assert cells == expected_cells
        assert valid_moves.dtype == np.int32
        assert playable_taboo.dtype == np.int32

    def test_empty_played_moves_uses_default_row_player_2(
        self, empty_taboo: np.ndarray, empty_played: np.ndarray
    ) -> None:
        """When Player 2 has no played moves, should return moves from last row."""
        board = self._create_board(4, empty_played, empty_played)

        node = self._create_node(
            game_board=board,
            current_player=2,
            played_p1=empty_played,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Player 2 should have access to all cells in last row (row 3)
        expected_cells = {(3, 0), (3, 1), (3, 2), (3, 3)}
        assert cells == expected_cells
        assert valid_moves.dtype == np.int32
        assert playable_taboo.dtype == np.int32

    def test_considers_neighbors_and_default_row_player_1(
        self, empty_taboo: np.ndarray, empty_played: np.ndarray
    ) -> None:
        """Player 1 moves should be from neighbors of played moves AND the first row."""
        played_p1 = np.array([[0, 0, 1]], dtype=np.int32)  # Corner position
        board = self._create_board(4, played_p1, empty_played)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)

        # Extract unique cells from result
        cells = set(map(tuple, all_moves[:, :2]))
        # Neighbors of (0,0) that are empty: (0,1), (1,0), (1,1)
        # Plus all empty cells in row 0: (0,1), (0,2), (0,3) - (0,0) is occupied
        expected_cells = {(0, 1), (0, 2), (0, 3), (1, 0), (1, 1)}

        assert cells == expected_cells

    def test_skips_occupied_cells(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Should not return moves for cells that are already occupied."""
        # Player 1 played at (0, 2), Player 2 played at (0, 3) - they are neighbors
        played_p1 = np.array([[0, 2, 2]], dtype=np.int32)
        played_p2 = np.array([[0, 3, 4]], dtype=np.int32)
        board = self._create_board(4, played_p1, played_p2)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=played_p2,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # (0, 3) is occupied by P2, should not be in result
        assert (0, 3) not in cells
        # (0, 1) is empty neighbor, should be in result
        assert (0, 1) in cells

    def test_valid_moves_keep_puzzle_solvable(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Valid moves should keep the puzzle solvable after placement."""
        played_p1 = np.array([[0, 0, 1]], dtype=np.int32)
        board = self._create_board(4, played_p1, empty_played, initial_values=[(0, 3, 4)])

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, _ = node._get_allowed_moves()

        # Every valid move should keep the puzzle solvable
        for move in valid_moves:
            r, c, val = int(move[0]), int(move[1]), int(move[2])
            test_board = board.copy()
            test_board[r, c] = val
            solution = solve(test_board, box_height=2, box_width=2)
            assert solution is not None, f"Valid move ({r},{c})->{val} should keep puzzle solvable"

    def test_taboo_moves_make_puzzle_unsolvable(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Taboo moves should make the puzzle unsolvable after placement."""
        played_p1 = np.array([[0, 0, 1]], dtype=np.int32)
        board = self._create_board(4, played_p1, empty_played, initial_values=[(0, 3, 4)])

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        _, playable_taboo = node._get_allowed_moves()

        # Every taboo move should make the puzzle unsolvable
        for move in playable_taboo:
            r, c, val = int(move[0]), int(move[1]), int(move[2])
            test_board = board.copy()
            test_board[r, c] = val
            solution = solve(test_board, box_height=2, box_width=2)
            assert solution is None, f"Taboo move ({r},{c})->{val} should make puzzle unsolvable"

    def test_excludes_already_proposed_taboo_moves(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Already proposed taboo moves should not appear in playable_taboo or valid_moves."""
        played_p1 = np.array([[0, 0, 1]], dtype=np.int32)
        board = self._create_board(4, played_p1, empty_played, initial_values=[(0, 3, 4)])

        # First get all moves without proposed taboo
        node_before = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )
        valid_before, taboo_before = node_before._get_allowed_moves()

        # Now propose a move as taboo (pick first valid move for testing)
        if len(valid_before) > 0:
            proposed_taboo = valid_before[:1]  # Take first valid move
        elif len(taboo_before) > 0:
            proposed_taboo = taboo_before[:1]  # Take first taboo move
        else:
            return  # No moves to test

        node_after = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=proposed_taboo,
            taboo_p2=empty_taboo,
        )
        valid_after, taboo_after = node_after._get_allowed_moves()

        # The proposed move should not appear in either result
        proposed_tuple = tuple(proposed_taboo[0])
        valid_set = set(map(tuple, valid_after))
        taboo_set = set(map(tuple, taboo_after))

        assert proposed_tuple not in valid_set, "Proposed taboo should not be in valid_moves"
        assert proposed_tuple not in taboo_set, "Proposed taboo should not be in playable_taboo"

    def test_player_1_uses_player_1_moves_and_first_row(self, empty_taboo: np.ndarray) -> None:
        """Player 1 should consider Player 1's played moves AND the first row."""
        played_p1 = np.array([[0, 0, 1]], dtype=np.int32)
        played_p2 = np.array([[3, 3, 3]], dtype=np.int32)
        board = self._create_board(4, played_p1, played_p2)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=played_p2,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Should have neighbors of (0, 0) and first row cells
        assert (0, 1) in cells  # neighbor and first row
        assert (0, 2) in cells  # first row
        assert (0, 3) in cells  # first row
        assert (1, 0) in cells or (1, 1) in cells  # neighbors of (0, 0)
        # Neighbors of (3, 3) should not be present (not first row, not P1 neighbors)
        assert (2, 2) not in cells
        assert (2, 3) not in cells
        assert (3, 2) not in cells

    def test_player_2_uses_player_2_moves_and_last_row(self, empty_taboo: np.ndarray) -> None:
        """Player 2 should consider Player 2's played moves AND the last row."""
        played_p1 = np.array([[0, 0, 1]], dtype=np.int32)
        played_p2 = np.array([[3, 3, 3]], dtype=np.int32)
        board = self._create_board(4, played_p1, played_p2)

        node = self._create_node(
            game_board=board,
            current_player=2,
            played_p1=played_p1,
            played_p2=played_p2,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Should have neighbors of (3, 3) and last row cells
        assert (3, 0) in cells  # last row
        assert (3, 1) in cells  # last row
        assert (3, 2) in cells  # neighbor and last row
        assert (2, 2) in cells or (2, 3) in cells  # neighbors of (3, 3)
        # Neighbors of (0, 0) should not be present (not last row, not P2 neighbors)
        assert (0, 1) not in cells
        assert (1, 0) not in cells
        assert (1, 1) not in cells

    def test_handles_multiple_played_moves(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Should correctly handle multiple played moves, their neighbors, and default row."""
        # Two moves: corner (0,0) and center area (1,2)
        played_p1 = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
        board = self._create_board(4, played_p1, empty_played)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Expected: neighbors of played moves + first row cells (excluding occupied)
        # Neighbors of (0,0) and (1,2): (0,1), (0,2), (0,3), (1,0), (1,1), (1,3), (2,1), (2,2), (2,3)
        # First row (excluding (0,0)): (0,1), (0,2), (0,3)
        # Combined (deduplicated): same set since first row cells are already in neighbors
        expected_cells = {(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3), (2, 1), (2, 2), (2, 3)}

        assert cells == expected_cells, (
            f"\nBoard:\n{board}\n"
            f"Played moves: {played_p1.tolist()}\n"
            f"Expected cells: {sorted(expected_cells)}\n"
            f"Actual cells: {sorted(cells)}\n"
        )

    def test_returns_correct_dtype(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Both result arrays should be int32 dtype."""
        played_p1 = np.array([[0, 0, 1]], dtype=np.int32)
        board = self._create_board(4, played_p1, empty_played)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()

        assert valid_moves.dtype == np.int32
        assert playable_taboo.dtype == np.int32

    def test_no_duplicate_moves_in_valid(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Each move in valid_moves should appear only once."""
        played_p1 = np.array([[0, 0, 1], [0, 2, 2]], dtype=np.int32)
        board = self._create_board(4, played_p1, empty_played)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, _ = node._get_allowed_moves()
        moves_set = set(map(tuple, valid_moves))

        assert len(moves_set) == len(valid_moves)

    def test_no_duplicate_moves_in_taboo(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Each move in playable_taboo should appear only once."""
        played_p1 = np.array([[0, 0, 1], [0, 2, 2]], dtype=np.int32)
        board = self._create_board(4, played_p1, empty_played)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        _, playable_taboo = node._get_allowed_moves()
        moves_set = set(map(tuple, playable_taboo))

        assert len(moves_set) == len(playable_taboo)

    def test_out_of_bounds_neighbors_filtered(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Neighbors outside board bounds should be filtered out."""
        played_p1 = np.array([[0, 0, 1]], dtype=np.int32)
        board = self._create_board(2, played_p1, empty_played)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
            box_height=1,
            box_width=2,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Only in-bounds empty neighbors
        for r, c in cells:
            assert 0 <= r < 2
            assert 0 <= c < 2

    def test_valid_and_taboo_are_disjoint(self, empty_taboo: np.ndarray, empty_played: np.ndarray) -> None:
        """Valid moves and taboo moves should not overlap for the same cell."""
        played_p1 = np.array([[0, 0, 1]], dtype=np.int32)
        board = self._create_board(4, played_p1, empty_played, initial_values=[(0, 3, 4)])

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()

        valid_set = set(map(tuple, valid_moves))
        taboo_set = set(map(tuple, playable_taboo))

        # No move should be in both valid and taboo
        intersection = valid_set & taboo_set
        assert len(intersection) == 0, f"Overlap found: {intersection}"

    def test_player_1_default_row_excludes_occupied_cells(
        self, empty_taboo: np.ndarray, empty_played: np.ndarray
    ) -> None:
        """Player 1's default row (row 0) should exclude occupied cells."""
        # Pre-fill some cells in row 0
        board = self._create_board(4, empty_played, empty_played, initial_values=[(0, 0, 1), (0, 2, 3)])

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=empty_played,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Only empty cells in row 0: (0,1) and (0,3)
        expected_cells = {(0, 1), (0, 3)}
        assert cells == expected_cells

    def test_player_2_default_row_excludes_occupied_cells(
        self, empty_taboo: np.ndarray, empty_played: np.ndarray
    ) -> None:
        """Player 2's default row (last row) should exclude occupied cells."""
        # Pre-fill some cells in row 3 (last row for 4x4 board)
        board = self._create_board(4, empty_played, empty_played, initial_values=[(3, 1, 2), (3, 3, 4)])

        node = self._create_node(
            game_board=board,
            current_player=2,
            played_p1=empty_played,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Only empty cells in row 3: (3,0) and (3,2)
        expected_cells = {(3, 0), (3, 2)}
        assert cells == expected_cells

    def test_player_1_combines_reachable_and_default_row(
        self, empty_taboo: np.ndarray, empty_played: np.ndarray
    ) -> None:
        """Player 1 should have access to both reachable cells and first row cells."""
        # Player 1 played in a non-first-row position
        played_p1 = np.array([[2, 2, 5]], dtype=np.int32)
        board = self._create_board(4, played_p1, empty_played)

        node = self._create_node(
            game_board=board,
            current_player=1,
            played_p1=played_p1,
            played_p2=empty_played,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Neighbors of (2,2): (1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)
        # Plus first row: (0,0), (0,1), (0,2), (0,3)
        expected_cells = {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),  # first row
            (1, 1),
            (1, 2),
            (1, 3),  # neighbors
            (2, 1),
            (2, 3),  # neighbors
            (3, 1),
            (3, 2),
            (3, 3),  # neighbors
        }
        assert cells == expected_cells

    def test_player_2_combines_reachable_and_default_row(
        self, empty_taboo: np.ndarray, empty_played: np.ndarray
    ) -> None:
        """Player 2 should have access to both reachable cells and last row cells."""
        # Player 2 played in a non-last-row position
        played_p2 = np.array([[1, 1, 5]], dtype=np.int32)
        board = self._create_board(4, empty_played, played_p2)

        node = self._create_node(
            game_board=board,
            current_player=2,
            played_p1=empty_played,
            played_p2=played_p2,
            taboo_p1=empty_taboo,
            taboo_p2=empty_taboo,
        )

        valid_moves, playable_taboo = node._get_allowed_moves()
        all_moves = self._combine_moves(valid_moves, playable_taboo)
        cells = set(map(tuple, all_moves[:, :2]))

        # Neighbors of (1,1): (0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)
        # Plus last row: (3,0), (3,1), (3,2), (3,3)
        expected_cells = {
            (0, 0),
            (0, 1),
            (0, 2),  # neighbors
            (1, 0),
            (1, 2),  # neighbors
            (2, 0),
            (2, 1),
            (2, 2),  # neighbors
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),  # last row
        }
        assert cells == expected_cells
