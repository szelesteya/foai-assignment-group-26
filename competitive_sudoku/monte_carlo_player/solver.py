import numpy as np


class SudokuBitmask:
    """Uses bitmasks for ultra-fast O(1) validation."""

    _row_masks: np.ndarray
    _col_masks: np.ndarray
    _box_masks: np.ndarray
    _n: int
    _box_height: int
    _box_width: int
    _full_mask: int

    def __init__(self, board: np.ndarray, box_height: int, box_width: int) -> None:
        self._n = board.shape[0]
        self._box_height = box_height
        self._box_width = box_width
        self._boxes_per_row = self._n // box_width

        # A complete region has bits 1 through N all set: (2^(N+1) - 1) - 1 = 2^(N+1) - 2
        # This creates a mask like 0b11111110 for N=7 (bits 1-7 set, bit 0 not set)
        self._full_mask = (1 << (self._n + 1)) - 2

        # Each mask is an int where bit i represents if number i is present
        self._row_masks = np.zeros(self._n, dtype=np.uint32)
        self._col_masks = np.zeros(self._n, dtype=np.uint32)
        self._box_masks = np.zeros(self._n, dtype=np.uint32)

        # Build masks from board
        for r in range(self._n):
            for c in range(self._n):
                val = board[r, c]
                if val != 0:
                    bit = 1 << val
                    self._row_masks[r] |= bit
                    self._col_masks[c] |= bit
                    self._box_masks[self._get_box_index(r, c)] |= bit

    def _get_box_index(self, row: int, col: int) -> int:
        return (row // self._box_height) * self._boxes_per_row + (col // self._box_width)

    def is_valid(self, row: int, col: int, num: int) -> bool:
        """Ultra-fast O(1) check using bitwise AND."""
        bit = 1 << num
        box_idx = self._get_box_index(row, col)
        # If bit is set in any mask, number already exists
        return not (self._row_masks[row] & bit or self._col_masks[col] & bit or self._box_masks[box_idx] & bit)

    def get_valid_numbers(self, row: int, col: int) -> list[int]:
        """Get all valid numbers for a cell in one operation."""
        box_idx = self._get_box_index(row, col)
        # Combine all constraints
        used = self._row_masks[row] | self._col_masks[col] | self._box_masks[box_idx]
        # Return numbers whose bits are NOT set
        return [num for num in range(1, self._n + 1) if not (used & (1 << num))]

    def place(self, row: int, col: int, num: int) -> None:
        """Place number - O(1)."""
        bit = 1 << num
        self._row_masks[row] |= bit
        self._col_masks[col] |= bit
        self._box_masks[self._get_box_index(row, col)] |= bit

    def remove(self, row: int, col: int, num: int) -> None:
        """Remove number (backtracking) - O(1)."""
        bit = 1 << num
        # Use XOR to unset the bit. Since we only call remove() after place(),
        # we know the bit is currently set.
        self._row_masks[row] ^= bit
        self._col_masks[col] ^= bit
        self._box_masks[self._get_box_index(row, col)] ^= bit

    def count_completed_regions(self, row: int, col: int, num: int) -> int:
        """Count how many regions would be completed by placing a number.

        A region (row, column, or block) is completed when all N numbers
        from 1 to N are placed in it.

        Args:
            row: The row index of the cell.
            col: The column index of the cell.
            num: The number to place.

        Returns:
            The number of regions (0, 1, 2, or 3) that would be completed.
        """
        bit = 1 << num
        box_idx = self._get_box_index(row, col)
        completed = 0

        # Check if row would be complete after placing this number
        if (self._row_masks[row] | bit) == self._full_mask:
            completed += 1

        # Check if column would be complete after placing this number
        if (self._col_masks[col] | bit) == self._full_mask:
            completed += 1

        # Check if box would be complete after placing this number
        if (self._box_masks[box_idx] | bit) == self._full_mask:
            completed += 1

        return completed

    def is_region_complete(self, region_type: str, index: int) -> bool:
        """Check if a specific region is already complete.

        Args:
            region_type: One of 'row', 'col', or 'box'.
            index: The index of the region.

        Returns:
            True if the region contains all numbers from 1 to N.
        """
        if region_type == "row":
            return self._row_masks[index] == self._full_mask
        elif region_type == "col":
            return self._col_masks[index] == self._full_mask
        elif region_type == "box":
            return self._box_masks[index] == self._full_mask
        else:
            raise ValueError(f"Invalid region_type: {region_type}. Must be 'row', 'col', or 'box'.")


def solve(board: np.ndarray, box_height: int, box_width: int) -> np.ndarray | None:
    """
    Solve a Sudoku board using backtracking with bitmask optimization.

    Args:
        board: The Sudoku board to solve.
        box_height: The height of the box.
        box_width: The width of the box.

    Returns:
        The solved board or None if no solution is found.
    """
    n = board.shape[0]
    sudoku = SudokuBitmask(board, box_height, box_width)
    board_copy = board.copy()

    def _backtrack() -> bool:
        # Find the first empty cell
        # Optimization: we could find the cell with fewest possibilities (MRV)
        # but for now let's just keep it simple but avoid copies.
        empty_positions = np.where(board_copy == 0)
        if len(empty_positions[0]) == 0:
            return True

        r, c = int(empty_positions[0][0]), int(empty_positions[1][0])
        available_numbers = sudoku.get_valid_numbers(r, c)

        for num in available_numbers:
            board_copy[r, c] = num
            sudoku.place(r, c, num)
            if _backtrack():
                return True
            # Backtrack
            sudoku.remove(r, c, num)
            board_copy[r, c] = 0

        return False

    if _backtrack():
        return board_copy
    return None


def main() -> None:
    """Test the Sudoku solver with sample puzzles and region completion counting."""
    print("=" * 60)
    print("REGION COMPLETION COUNTING TEST")
    print("=" * 60)

    # 4x4 board almost complete - perfect for testing region completion
    # Missing only one cell in row 0, col 3, and box 1
    board_almost_complete = np.array(
        [
            [1, 2, 3, 0],  # Row 0 missing 4, Col 3 missing 4, Box 1 missing 4
            [3, 4, 1, 2],
            [4, 3, 2, 1],
            [2, 1, 4, 3],
        ],
        dtype=np.int32,
    )

    print("\nBoard (almost complete - missing one cell at (0,3)):")
    for row in board_almost_complete:
        print("  " + " ".join(str(v) if v != 0 else "." for v in row))

    sudoku = SudokuBitmask(board_almost_complete, box_height=2, box_width=2)

    # Placing 4 at (0,3) should complete row 0, col 3, and box 1 (top-right)
    regions_completed = sudoku.count_completed_regions(0, 3, 4)
    print("\nPlacing 4 at (0, 3):")
    print(f"  Regions completed: {regions_completed}")
    print("  Expected: 3 (row 0, col 3, box 1)")
    print(f"  ✓ Correct: {regions_completed == 3}")

    print("\n" + "-" * 60)

    # Test with a board where only one region is completed
    board_one_region = np.array(
        [
            [1, 2, 3, 0],  # Row 0 will be complete with 4
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    print("\nBoard (only row 0 almost complete):")
    for row in board_one_region:
        print("  " + " ".join(str(v) if v != 0 else "." for v in row))

    sudoku2 = SudokuBitmask(board_one_region, box_height=2, box_width=2)
    regions_completed = sudoku2.count_completed_regions(0, 3, 4)
    print("\nPlacing 4 at (0, 3):")
    print(f"  Regions completed: {regions_completed}")
    print("  Expected: 1 (only row 0)")
    print(f"  ✓ Correct: {regions_completed == 1}")

    print("\n" + "-" * 60)

    # Test with board where two regions are completed
    board_two_regions = np.array(
        [
            [1, 2, 3, 0],  # Row 0 complete with 4
            [0, 0, 0, 4],
            [0, 0, 0, 3],
            [0, 0, 0, 2],  # Col 3 complete with 1 at (3,3)? No, let's fix
        ],
        dtype=np.int32,
    )

    # Actually let's make col 3 almost complete
    board_two_regions = np.array(
        [
            [1, 2, 3, 0],  # Row 0 needs 4, Col 3 needs 4
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 3],
        ],
        dtype=np.int32,
    )

    print("\nBoard (row 0 and col 3 almost complete):")
    for row in board_two_regions:
        print("  " + " ".join(str(v) if v != 0 else "." for v in row))

    sudoku3 = SudokuBitmask(board_two_regions, box_height=2, box_width=2)
    regions_completed = sudoku3.count_completed_regions(0, 3, 4)
    print("\nPlacing 4 at (0, 3):")
    print(f"  Regions completed: {regions_completed}")
    print("  Expected: 2 (row 0 and col 3)")
    print(f"  ✓ Correct: {regions_completed == 2}")

    print("\n" + "=" * 60)
    print("SUDOKU SOLVER TEST")
    print("=" * 60)

    # Standard 9x9 Sudoku (3x3 boxes)
    board_9x9 = np.array(
        [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]
    )

    print("\nOriginal 9x9 board:")
    print(board_9x9)

    solution = solve(board_9x9, box_height=3, box_width=3)

    if solution is not None:
        print("\nSolved 9x9 board:")
        print(solution)
    else:
        print("\nNo solution found for 9x9 board.")

    print("\n" + "-" * 60)

    # 4x4 Sudoku (2x2 boxes)
    board_4x4 = np.array(
        [
            [1, 0, 0, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 3],
        ]
    )

    print("\nOriginal 4x4 board:")
    print(board_4x4)

    solution = solve(board_4x4, box_height=2, box_width=2)

    if solution is not None:
        print("\nSolved 4x4 board:")
        print(solution)
    else:
        print("\nNo solution found for 4x4 board.")


if __name__ == "__main__":
    main()
