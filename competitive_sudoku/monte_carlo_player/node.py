from __future__ import annotations

from typing import override

import numpy as np

from .mcts import Node
from .solver import SudokuBitmask, solve

P_TABOO_MOVE_IN_SIMULATION = 0.1
P_RANDOM_MOVE_IN_SIMULATION = 0.3
P_HEURISTIC_MOVE_IN_SIMULATION = 0.6


class SudokuNode(Node):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    # Scoring table: maps number of completed regions to points scored
    # 0 regions -> 0 points, 1 region -> 1 point, 2 regions -> 3 points, 3 regions -> 7 points
    _REGION_SCORE_TABLE: tuple[int, ...] = (0, 1, 3, 7)

    __slots__ = (
        "game_board",
        "current_player",
        "played_moves_player1",
        "played_moves_player2",
        "player_1_score",
        "player_2_score",
        "proposed_taboo_moves_player1",
        "proposed_taboo_moves_player2",
        "box_height",
        "box_width",
    )

    def __init__(
        self,
        game_board: np.ndarray,
        current_player: int,
        played_moves_player1: np.ndarray,
        played_moves_player2: np.ndarray,
        player_1_score: int,
        player_2_score: int,
        proposed_taboo_moves_player1: np.ndarray,
        proposed_taboo_moves_player2: np.ndarray,
        box_height: int,
        box_width: int,
    ):
        self.game_board = game_board
        self.current_player = current_player
        self.played_moves_player1 = played_moves_player1
        self.player_1_score = player_1_score
        self.played_moves_player2 = played_moves_player2
        self.player_2_score = player_2_score
        self.proposed_taboo_moves_player1 = proposed_taboo_moves_player1
        self.proposed_taboo_moves_player2 = proposed_taboo_moves_player2
        self.box_height = box_height
        self.box_width = box_width

    # Pre-computed 8-neighbor offsets (excludes center cell)
    _NEIGHBOR_OFFSETS: np.ndarray = np.array(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int32
    )

    def _get_reachable_cells(self, player_id: int) -> np.ndarray:
        """Get all empty cells reachable by the specified player.

        Reachable cells are neighbors of already played moves by that player,
        plus the default row (row 0 for player 1, last row for player 2).
        This does NOT check if any valid Sudoku value can be placed there.

        Args:
            player_id: The ID of the player (1 or 2).

        Returns:
            An ndarray of shape (n, 2) with [row, col] of reachable empty cells.
        """
        is_p1 = player_id == 1
        played = self.played_moves_player1 if is_p1 else self.played_moves_player2
        n = self.game_board.shape[0]

        # Get default row cells for current player:
        # Player 1 -> first row (row 0), Player 2 -> last row (row n-1)
        default_row = 0 if is_p1 else n - 1
        default_row_cells = np.array([[default_row, c] for c in range(n)], dtype=np.int32)
        # Filter to only empty cells in the default row
        default_row_cells = default_row_cells[self.game_board[default_row_cells[:, 0], default_row_cells[:, 1]] == 0]

        if len(played) == 0:
            return default_row_cells

        # Vectorized: compute all neighbors, filter in-bounds, dedupe, filter empty
        positions = played[:, :2].astype(np.int32)
        neighbors = (positions[:, np.newaxis, :] + self._NEIGHBOR_OFFSETS[np.newaxis, :, :]).reshape(-1, 2)
        mask = (neighbors >= 0).all(axis=1) & (neighbors < n).all(axis=1)
        reachable_cells = np.unique(neighbors[mask], axis=0)
        reachable_cells = reachable_cells[self.game_board[reachable_cells[:, 0], reachable_cells[:, 1]] == 0]

        # Combine reachable cells with default row cells and remove duplicates
        if len(reachable_cells) > 0 and len(default_row_cells) > 0:
            return np.unique(np.vstack([reachable_cells, default_row_cells]), axis=0)
        elif len(reachable_cells) > 0:
            return reachable_cells
        else:
            return default_row_cells

    def _find_single_holes(self) -> np.ndarray:
        """Find all squares that are the only empty square in their row, column, or box.

        Returns:
            An ndarray of shape (n, 2) with [row, col] of single holes.
        """
        n = self.game_board.shape[0]
        holes = []
        sudoku = SudokuBitmask(self.game_board, self.box_height, self.box_width)

        # Check rows
        for r in range(n):
            mask = sudoku._row_masks[r]
            missing = sudoku._full_mask ^ mask
            # If exactly one bit is missing (missing is power of 2)
            if missing > 0 and (missing & (missing - 1)) == 0:
                c = np.where(self.game_board[r, :] == 0)[0]
                if len(c) == 1:
                    holes.append([r, c[0]])

        # Check cols
        for c in range(n):
            mask = sudoku._col_masks[c]
            missing = sudoku._full_mask ^ mask
            if missing > 0 and (missing & (missing - 1)) == 0:
                r = np.where(self.game_board[:, c] == 0)[0]
                if len(r) == 1:
                    holes.append([r[0], c])

        # Check boxes
        for b in range(n):
            mask = sudoku._box_masks[b]
            missing = sudoku._full_mask ^ mask
            if missing > 0 and (missing & (missing - 1)) == 0:
                bh, bw = self.box_height, self.box_width
                br = (b // (n // bw)) * bh
                bc = (b % (n // bw)) * bw
                box = self.game_board[br : br + bh, bc : bc + bw]
                empty_idx = np.where(box == 0)
                if len(empty_idx[0]) == 1:
                    holes.append([br + empty_idx[0][0], bc + empty_idx[1][0]])

        if not holes:
            return np.empty((0, 2), dtype=np.int32)
        return np.unique(np.array(holes, dtype=np.int32), axis=0)

    def _calculate_potential_points(self, row: int, col: int) -> int:
        """Calculate the points that would be gained by filling a single hole.

        Args:
            row: The row index.
            col: The column index.

        Returns:
            The points (1, 3, or 7) that would be gained.
        """
        sudoku = SudokuBitmask(self.game_board, self.box_height, self.box_width)
        valid_vals = sudoku.get_valid_numbers(row, col)
        if not valid_vals:
            return 0
        # For a single hole, we take the first valid value to estimate points
        val = valid_vals[0]
        completed = sudoku.count_completed_regions(row, col, val)
        return self._REGION_SCORE_TABLE[completed]

    def evaluate(self, initial_opp_reach: np.ndarray | None = None) -> float:
        """Heuristic evaluation of the board state.

        Replicates the logic from team26_A2/sudokuai.py:170-218.
        Evaluates from the perspective of the player who just moved
        (i.e., the opponent of self.current_player).

        Args:
            initial_opp_reach: Optional reach of the opponent at the start of the turn.
                               Used to calculate the denial bonus.

        Returns:
            The heuristic score for the current board state.
        """
        # perspective: player who just moved
        my_id = 2 if self.current_player == 1 else 1
        opp_id = self.current_player

        # --- 1. Score Difference ---
        my_score = self.player_2_score if my_id == 2 else self.player_1_score
        opp_score = self.player_1_score if my_id == 2 else self.player_2_score
        real_score_diff = my_score - opp_score

        # --- 2. Mobility & Denial ---
        my_reach = self._get_reachable_cells(my_id)
        opp_reach_current = self._get_reachable_cells(opp_id)

        # Convert to sets for O(1) membership testing
        my_reach_set = {tuple(p) for p in my_reach}
        opp_reach_set = {tuple(p) for p in opp_reach_current}

        # Mobility Score: "How many moves do I have vs him?"
        # Heavily penalize opponent moves (Denial Strategy)
        mobility_score = len(my_reach) - (3.0 * len(opp_reach_current))

        # Gateway Protection Score (Heuristic approximation):
        # Check if we successfully reduced the opponent's reach compared to start of turn
        denial_bonus = 0.0
        if initial_opp_reach is not None:
            denial_bonus = float(len(initial_opp_reach) - len(opp_reach_current))

        # --- 3. Pending Points (Trap) ---
        pending_score = 0
        single_holes = self._find_single_holes()
        for i in range(len(single_holes)):
            r, c = int(single_holes[i, 0]), int(single_holes[i, 1])
            can_me = (r, c) in my_reach_set
            can_opp = (r, c) in opp_reach_set
            pts = self._calculate_potential_points(r, c)

            if can_opp:
                pending_score -= pts
            elif can_me:
                pending_score += pts

        # --- Weights ---
        W_SCORE = 20000.0
        W_PENDING = 15000.0
        W_MOBILITY = 200.0
        W_DENIAL_BONUS = 500.0

        total = (
            (real_score_diff * W_SCORE)
            + (pending_score * W_PENDING)
            + (mobility_score * W_MOBILITY)
            + (denial_bonus * W_DENIAL_BONUS)
        )

        return float(total)

    def _get_allowed_moves(self, check_solvability: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Get valid moves and playable taboo moves for the current player.

        A move is valid if:
        - It follows Sudoku constraints (row, column, box)
        - The puzzle remains solvable after placing it (if check_solvability is True)

        A move is a playable taboo if:
        - It follows Sudoku constraints (no immediate conflict)
        - BUT the puzzle becomes unsolvable after placing it
        - It hasn't been proposed as taboo yet

        Args:
            check_solvability: If True, calls the solver to check if the move keeps the puzzle solvable.

        Returns:
            A tuple of two ndarrays:
            - valid_moves: shape (n, 3) with [row, col, value] for truly valid moves.
            - playable_taboo: shape (m, 3) with [row, col, value] for moves that make puzzle unsolvable.
        """
        empty = np.empty((0, 3), dtype=np.int32)
        is_p1 = self.current_player == 1
        taboo = self.proposed_taboo_moves_player1 if is_p1 else self.proposed_taboo_moves_player2

        cells = self._get_reachable_cells(self.current_player)

        if len(cells) == 0:
            return empty, empty

        sudoku = SudokuBitmask(self.game_board, self.box_height, self.box_width)
        taboo_set = {tuple(m) for m in taboo.astype(int)}

        valid_moves: list[list[int]] = []
        playable_taboo: list[list[int]] = []

        for r, c in cells:
            # Get values that don't violate immediate Sudoku constraints
            candidate_values = sudoku.get_valid_numbers(r, c)

            for val in candidate_values:
                # Skip if already proposed as taboo
                if (r, c, val) in taboo_set:
                    continue

                if not check_solvability:
                    valid_moves.append([r, c, val])
                    continue

                # Try placing the value and check if puzzle is still solvable
                board_copy = self.game_board.copy()
                board_copy[r, c] = val
                solution = solve(board_copy, self.box_height, self.box_width)

                if solution is not None:
                    # Puzzle is solvable -> valid move
                    valid_moves.append([r, c, val])
                else:
                    # Puzzle becomes unsolvable -> taboo move
                    playable_taboo.append([r, c, val])

        valid_arr = np.array(valid_moves, dtype=np.int32) if valid_moves else empty
        taboo_arr = np.array(playable_taboo, dtype=np.int32) if playable_taboo else empty

        return valid_arr, taboo_arr

    def _create_child_node(
        self,
        move: np.ndarray,
        is_taboo_move: bool,
    ) -> "SudokuNode":
        """Create a child node after playing a move.

        Args:
            move: The move to play as [row, col, value].
            is_taboo_move: Whether this move is a taboo move (makes puzzle unsolvable).

        Returns:
            A new SudokuNode representing the state after the move.
        """
        row, col, value = int(move[0]), int(move[1]), int(move[2])
        next_player = 2 if self.current_player == 1 else 1

        # Calculate score for this move based on completed regions
        # Only valid moves (non-taboo) score points
        new_p1_score = self.player_1_score
        new_p2_score = self.player_2_score

        if not is_taboo_move:
            # Use SudokuBitmask to count completed regions before placing the move
            sudoku = SudokuBitmask(self.game_board, self.box_height, self.box_width)
            completed_regions = sudoku.count_completed_regions(row, col, value)
            points_scored = self._REGION_SCORE_TABLE[completed_regions]

            if self.current_player == 1:
                new_p1_score += points_scored
            else:
                new_p2_score += points_scored

        # Update the game board with the played move
        new_board = self.game_board.copy()
        new_board[row, col] = value

        # Update played moves for the current player
        move_entry = np.array([[row, col, value]], dtype=np.int32)
        if self.current_player == 1:
            new_played_p1 = np.vstack([self.played_moves_player1, move_entry])
            new_played_p2 = self.played_moves_player2.copy()
        else:
            new_played_p1 = self.played_moves_player1.copy()
            new_played_p2 = np.vstack([self.played_moves_player2, move_entry])

        # Update taboo moves if this is a taboo move
        if is_taboo_move:
            if self.current_player == 1:
                new_taboo_p1 = np.vstack([self.proposed_taboo_moves_player1, move_entry])
                new_taboo_p2 = self.proposed_taboo_moves_player2.copy()
            else:
                new_taboo_p1 = self.proposed_taboo_moves_player1.copy()
                new_taboo_p2 = np.vstack([self.proposed_taboo_moves_player2, move_entry])
        else:
            new_taboo_p1 = self.proposed_taboo_moves_player1.copy()
            new_taboo_p2 = self.proposed_taboo_moves_player2.copy()

        return SudokuNode(
            game_board=new_board,
            current_player=next_player,
            played_moves_player1=new_played_p1,
            played_moves_player2=new_played_p2,
            player_1_score=new_p1_score,
            player_2_score=new_p2_score,
            proposed_taboo_moves_player1=new_taboo_p1,
            proposed_taboo_moves_player2=new_taboo_p2,
            box_height=self.box_height,
            box_width=self.box_width,
        )

    def _create_pass_turn_child(self) -> "SudokuNode":
        """Create a child node that passes the turn to the other player.

        Used when the current player has no valid moves available.
        The board state remains unchanged, only the current player switches.

        Returns:
            A new SudokuNode with the turn passed to the other player.
        """
        next_player = 2 if self.current_player == 1 else 1

        return SudokuNode(
            game_board=self.game_board.copy(),
            current_player=next_player,
            played_moves_player1=self.played_moves_player1.copy(),
            played_moves_player2=self.played_moves_player2.copy(),
            player_1_score=self.player_1_score,
            player_2_score=self.player_2_score,
            proposed_taboo_moves_player1=self.proposed_taboo_moves_player1.copy(),
            proposed_taboo_moves_player2=self.proposed_taboo_moves_player2.copy(),
            box_height=self.box_height,
            box_width=self.box_width,
        )

    @override
    def find_children(self) -> set["SudokuNode"]:
        """Generate all possible successor states of this board.

        Creates child nodes for both valid moves and taboo moves.
        Each child has the move applied to the board and the player switched.

        If the current player has no valid moves, the turn is passed to the
        other player (creates a single child with the same board state but
        switched player).
        """
        valid_moves, playable_taboo = self._get_allowed_moves(check_solvability=True)
        children: set[SudokuNode] = set()

        # If no moves available, pass the turn to the other player
        if len(valid_moves) == 0 and len(playable_taboo) == 0:
            pass_child = self._create_pass_turn_child()
            children.add(pass_child)
            return children

        # Create children for valid moves
        for move in valid_moves:
            child = self._create_child_node(move, is_taboo_move=False)
            children.add(child)

        # Create children for taboo moves as well
        for move in playable_taboo:
            child = self._create_child_node(move, is_taboo_move=True)
            children.add(child)

        return children

    @override
    def find_random_child(self) -> SudokuNode:
        """Random successor of this board state (for more efficient simulation).

        During simulation, we prioritize speed and only check basic Sudoku constraints
        without running the full solver for every move.
        """
        valid_moves, _ = self._get_allowed_moves(check_solvability=False)

        # If no moves available, try passing the turn
        if len(valid_moves) == 0:
            return self._create_pass_turn_child()

        # Pick a random valid move
        move_idx = np.random.randint(len(valid_moves))
        return self._create_child_node(valid_moves[move_idx], is_taboo_move=False)

    @override
    def is_terminal(self) -> bool:
        """Check if the state is terminal (no more moves possible for either player)."""
        # Simple check: if board is full, it's terminal
        if np.all(self.game_board != 0):
            return True

        # If current player has moves, it's not terminal
        # (Fast check during simulation, doesn't need full solvability check)
        valid, _ = self._get_allowed_moves(check_solvability=False)
        if len(valid) > 0:
            return False

        # If current player must pass, check if the other player also has no moves
        pass_child = self._create_pass_turn_child()
        v2, _ = pass_child._get_allowed_moves(check_solvability=False)
        return len(v2) == 0

    @override
    def reward(self) -> float:
        """Calculate the reward based on the final scores.

        Assumes `self` is a terminal node. Returns the reward from the
        perspective of player 1:
        - 1.0 if player 1 wins (higher score)
        - 0.0 if player 2 wins (higher score)
        - 0.5 if it's a tie (equal scores)

        Returns:
            The reward value (1.0 for win, 0.0 for loss, 0.5 for tie).
        """
        if self.player_1_score > self.player_2_score:
            return 1.0  # Player 1 wins
        elif self.player_1_score < self.player_2_score:
            return 0.0  # Player 2 wins
        else:
            return 0.5  # Tie

    @override
    def __hash__(self) -> int:
        """Hash based on board state and current player.

        Uses the game board bytes and current player for efficient hashing.
        Two nodes with the same board state and player will have the same hash.
        """
        return hash((self.game_board.tobytes(), self.current_player))

    @override
    def __eq__(self, other: object) -> bool:
        """Check equality based on board state, player, and move history.

        Two nodes are equal if they have:
        - Same game board
        - Same current player
        - Same played moves for both players
        - Same proposed taboo moves for both players
        - Same scores for both players
        """
        if not isinstance(other, SudokuNode):
            return False
        return (
            np.array_equal(self.game_board, other.game_board)
            and self.current_player == other.current_player
            and np.array_equal(self.played_moves_player1, other.played_moves_player1)
            and np.array_equal(self.played_moves_player2, other.played_moves_player2)
            and np.array_equal(self.proposed_taboo_moves_player1, other.proposed_taboo_moves_player1)
            and np.array_equal(self.proposed_taboo_moves_player2, other.proposed_taboo_moves_player2)
            and self.player_1_score == other.player_1_score
            and self.player_2_score == other.player_2_score
        )


def _print_moves(moves: np.ndarray, label: str) -> None:
    """Helper to print moves grouped by cell."""
    if len(moves) == 0:
        print(f"  No {label}")
        return
    cells = sorted(set(map(tuple, moves[:, :2])))
    for cell in cells:
        cell_moves = moves[(moves[:, 0] == cell[0]) & (moves[:, 1] == cell[1])]
        values = sorted(cell_moves[:, 2])
        print(f"  ({cell[0]}, {cell[1]}) -> {values}")


def _print_board(board: np.ndarray) -> None:
    """Helper to print a board with nice formatting."""
    for row in board:
        print("  " + " ".join(str(v) if v != 0 else "." for v in row))


def main() -> None:
    """Test the SudokuNode class capabilities including child node generation."""
    print("=" * 70)
    print("SUDOKU NODE CAPABILITIES TEST")
    print("=" * 70)

    # Create a 4x4 board where some moves will make it unsolvable
    board = np.array(
        [
            [1, 0, 0, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 3],
        ],
        dtype=np.int32,
    )

    # First, show what the solution looks like
    print("\n[1] BOARD SETUP")
    print("-" * 40)
    print("Original board:")
    _print_board(board)
    solution = solve(board.copy(), 2, 2)
    print("\nUnique solution:")
    _print_board(solution)

    # Player 1 played at (0,0)=1
    played_p1 = np.array([[0, 0, 1]], dtype=np.int32)
    empty_played = np.empty((0, 3), dtype=np.int32)
    empty_taboo = np.empty((0, 3), dtype=np.int32)

    node = SudokuNode(
        game_board=board,
        current_player=1,
        played_moves_player1=played_p1,
        played_moves_player2=empty_played,
        player_1_score=0,
        player_2_score=0,
        proposed_taboo_moves_player1=empty_taboo,
        proposed_taboo_moves_player2=empty_taboo,
        box_height=2,
        box_width=2,
    )

    # Test _get_allowed_moves
    print("\n[2] MOVE DETECTION (_get_allowed_moves)")
    print("-" * 40)
    print(f"Player 1 played at: {played_p1.tolist()}")
    print("Analyzing neighboring cells...")

    valid_moves, taboo_moves = node._get_allowed_moves()

    print(f"\n✓ VALID MOVES ({len(valid_moves)} total) - puzzle remains solvable:")
    _print_moves(valid_moves, "valid moves")

    print(f"\n✗ TABOO MOVES ({len(taboo_moves)} total) - would make puzzle UNSOLVABLE:")
    _print_moves(taboo_moves, "taboo moves")

    # Test find_children
    print("\n[3] CHILD NODE GENERATION (find_children)")
    print("-" * 40)

    children = node.find_children()
    expected_children = len(valid_moves) + len(taboo_moves)

    print(f"Expected children: {expected_children} (valid: {len(valid_moves)}, taboo: {len(taboo_moves)})")
    print(f"Generated children: {len(children)}")
    print(f"✓ Count matches: {len(children) == expected_children}")

    # Test child node properties
    print("\n[4] CHILD NODE PROPERTIES")
    print("-" * 40)

    # Convert children to list for easier inspection
    children_list = list(children)

    if len(children_list) > 0:
        # Pick first child and inspect
        child = children_list[0]

        print("\nParent state:")
        print(f"  Current player: {node.current_player}")
        print(f"  Player 1 moves: {node.played_moves_player1.tolist()}")
        print(f"  Player 1 taboo: {node.proposed_taboo_moves_player1.tolist()}")
        print(f"  Scores - P1: {node.player_1_score}, P2: {node.player_2_score}")

        print("\nChild state (example):")
        print(f"  Current player: {child.current_player}")
        print(f"  Player switched: {child.current_player != node.current_player}")
        print(f"  Player 1 moves: {child.played_moves_player1.tolist()}")
        print(f"  Player 1 taboo: {child.proposed_taboo_moves_player1.tolist()}")
        print(f"  Scores - P1: {child.player_1_score}, P2: {child.player_2_score}")

        # Find which move was made
        if len(child.played_moves_player1) > len(node.played_moves_player1):
            new_move = child.played_moves_player1[-1]
            is_taboo_child = len(child.proposed_taboo_moves_player1) > len(node.proposed_taboo_moves_player1)
        else:
            new_move = child.played_moves_player2[-1]
            is_taboo_child = len(child.proposed_taboo_moves_player2) > len(node.proposed_taboo_moves_player2)

        r, c, v = int(new_move[0]), int(new_move[1]), int(new_move[2])
        print(f"\n  Move played: ({r}, {c}) -> {v}")
        print(f"  Is taboo move: {is_taboo_child}")
        print(f"  Board value at ({r}, {c}): {child.game_board[r, c]}")
        print(f"  ✓ Move applied correctly: {child.game_board[r, c] == v}")

    # Verify all children have moves applied
    print("\n[5] VERIFY ALL CHILDREN")
    print("-" * 40)

    all_valid = True
    valid_children = 0
    taboo_children = 0

    for child in children:
        # Check player switched
        if child.current_player == node.current_player:
            print("  ✗ Player not switched!")
            all_valid = False
            continue

        # Find the new move
        if len(child.played_moves_player1) > len(node.played_moves_player1):
            new_move = child.played_moves_player1[-1]
            is_taboo = len(child.proposed_taboo_moves_player1) > len(node.proposed_taboo_moves_player1)
        else:
            new_move = child.played_moves_player2[-1] if len(child.played_moves_player2) > 0 else None
            is_taboo = len(child.proposed_taboo_moves_player2) > len(node.proposed_taboo_moves_player2)

        if new_move is None:
            print("  ✗ No new move found!")
            all_valid = False
            continue

        r, c, v = int(new_move[0]), int(new_move[1]), int(new_move[2])

        # Check move applied to board
        if child.game_board[r, c] != v:
            print(f"  ✗ Move ({r}, {c}) -> {v} not applied to board!")
            all_valid = False
            continue

        if is_taboo:
            taboo_children += 1
        else:
            valid_children += 1

    print(f"Valid move children: {valid_children} (expected: {len(valid_moves)})")
    print(f"Taboo move children: {taboo_children} (expected: {len(taboo_moves)})")
    print(f"✓ All children valid: {all_valid}")
    print(f"✓ Valid count matches: {valid_children == len(valid_moves)}")
    print(f"✓ Taboo count matches: {taboo_children == len(taboo_moves)}")

    # Test multi-level tree generation
    print("\n[6] MULTI-LEVEL TREE (2 levels deep)")
    print("-" * 40)

    if len(children_list) > 0:
        # Pick a valid move child (non-taboo) for cleaner demonstration
        valid_child = None
        for child in children_list:
            if len(child.proposed_taboo_moves_player1) == len(node.proposed_taboo_moves_player1):
                valid_child = child
                break

        if valid_child is not None:
            print(f"\nLevel 1 child (Player {valid_child.current_player}'s turn):")
            print("  Board:")
            _print_board(valid_child.game_board)

            grandchildren = valid_child.find_children()
            print(f"\n  Level 2 children (grandchildren): {len(grandchildren)}")

            if len(grandchildren) > 0:
                grandchild = list(grandchildren)[0]
                print(f"\n  Example grandchild (Player {grandchild.current_player}'s turn):")
                print("    Board:")
                for row in grandchild.game_board:
                    print("      " + " ".join(str(v) if v != 0 else "." for v in row))
                print(f"    Player 1 moves: {len(grandchild.played_moves_player1)}")
                print(f"    Player 2 moves: {len(grandchild.played_moves_player2)}")
        else:
            print("  No valid (non-taboo) children found for demonstration")

    # Test scoring mechanism
    print("\n[7] SCORING TEST (Region Completion)")
    print("-" * 40)

    # Create a board where a move completes multiple regions
    # This board is almost complete, missing only position (0, 3)
    # Placing 4 at (0, 3) should complete row 0, col 3, and box 1 (top-right)
    scoring_board = np.array(
        [
            [1, 2, 3, 0],  # Row 0 missing 4
            [3, 4, 1, 2],
            [4, 3, 2, 1],
            [2, 1, 4, 3],
        ],
        dtype=np.int32,
    )

    print("Testing board (almost complete - one cell missing):")
    _print_board(scoring_board)

    # Player 1's last move was at (0, 2)=3, so (0, 3) is a neighbor
    scoring_played_p1 = np.array([[0, 0, 1], [0, 1, 2], [0, 2, 3]], dtype=np.int32)

    scoring_node = SudokuNode(
        game_board=scoring_board,
        current_player=1,
        played_moves_player1=scoring_played_p1,
        played_moves_player2=empty_played,
        player_1_score=5,  # Already has some score
        player_2_score=3,
        proposed_taboo_moves_player1=empty_taboo,
        proposed_taboo_moves_player2=empty_taboo,
        box_height=2,
        box_width=2,
    )

    print(f"\nParent scores - P1: {scoring_node.player_1_score}, P2: {scoring_node.player_2_score}")

    # Get children and check if scoring is applied
    scoring_children = list(scoring_node.find_children())
    print(f"\nGenerated {len(scoring_children)} children")

    # Find the child where 4 was placed at (0, 3)
    for child in scoring_children:
        if len(child.played_moves_player1) > len(scoring_played_p1):
            new_move = child.played_moves_player1[-1]
            r, c, v = int(new_move[0]), int(new_move[1]), int(new_move[2])
            if r == 0 and c == 3 and v == 4:
                print(f"\nMove ({r}, {c}) -> {v} (completes 3 regions):")
                print("  Expected score: P1: 5 + 7 = 12 (3 regions = 7 points)")
                print(f"  Actual scores - P1: {child.player_1_score}, P2: {child.player_2_score}")
                print(f"  ✓ Score correct: {child.player_1_score == 12}")
                print(f"  Reward (P1 perspective): {child.reward()}")
                break

    # Test playing a sequence of moves using find_random_child
    print("\n[8] SIMULATED PLAY (find_random_child)")
    print("-" * 40)

    current_sim_node = node
    print("Starting simulation from original board...")

    for i in range(5):
        if current_sim_node.is_terminal():
            print(f"Reached terminal state at step {i}")
            break

        next_sim_node = current_sim_node.find_random_child()
        if next_sim_node is None:
            print(f"No child found at step {i}")
            break

        # Determine the move made
        is_p1 = current_sim_node.current_player == 1
        old_moves = current_sim_node.played_moves_player1 if is_p1 else current_sim_node.played_moves_player2
        new_moves = next_sim_node.played_moves_player1 if is_p1 else next_sim_node.played_moves_player2

        if len(new_moves) > len(old_moves):
            move = new_moves[-1]
            r, c, v = int(move[0]), int(move[1]), int(move[2])
            print(f"Step {i+1}: Player {current_sim_node.current_player} played ({r}, {c}) -> {v}")
        else:
            print(f"Step {i+1}: Player {current_sim_node.current_player} passed turn")

        current_sim_node = next_sim_node

    print("\nFinal board after 5 steps:")
    _print_board(current_sim_node.game_board)
    print(f"Final scores - P1: {current_sim_node.player_1_score}, P2: {current_sim_node.player_2_score}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
