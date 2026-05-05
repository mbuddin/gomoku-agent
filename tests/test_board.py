import unittest
from gomoku.board import Board, EMPTY, BLACK, WHITE

class TestBoard(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_initial_state(self):
        """Ensure the board starts completely empty."""
        self.assertEqual(self.board.stone_count, 0)
        self.assertFalse(self.board.is_full())
        self.assertEqual(len(self.board.get_empty_cells()), 15 * 15)

    def test_place_and_remove_stone(self):
        """Test placing and removing a stone updates the grid and count correctly."""
        self.board.place_stone(7, 7, BLACK)
        self.assertEqual(self.board.grid[7][7], BLACK)
        self.assertEqual(self.board.stone_count, 1)
        self.assertFalse(self.board.is_valid_move(7, 7))

        self.board.remove_stone(7, 7)
        self.assertEqual(self.board.grid[7][7], EMPTY)
        self.assertEqual(self.board.stone_count, 0)
        self.assertTrue(self.board.is_valid_move(7, 7))

    def test_out_of_bounds(self):
        """Ensure out-of-bounds moves raise errors and return False."""
        self.assertFalse(self.board.in_bounds(-1, 5))
        self.assertFalse(self.board.in_bounds(15, 15))

        with self.assertRaises(ValueError):
            self.board.place_stone(-1, 5, BLACK)

    def test_occupied_cell_error(self):
        """Ensure placing a stone on an occupied cell raises an error."""
        self.board.place_stone(0, 0, WHITE)
        with self.assertRaises(ValueError):
            self.board.place_stone(0, 0, BLACK)

    def test_horizontal_win(self):
        """Test a standard 5-in-a-row win horizontally."""
        for col in range(4):
            self.board.place_stone(0, col, BLACK)
            self.assertFalse(self.board.check_win(0, col, BLACK))
        
        # The 5th stone should trigger the win
        self.board.place_stone(0, 4, BLACK)
        self.assertTrue(self.board.check_win(0, 4, BLACK))

    def test_diagonal_win(self):
        """Test a 5-in-a-row win diagonally."""
        for i in range(4):
            self.board.place_stone(i, i, WHITE)
            self.assertFalse(self.board.check_win(i, i, WHITE))
            
        self.board.place_stone(4, 4, WHITE)
        self.assertTrue(self.board.check_win(4, 4, WHITE))

    def test_copy_board(self):
        """Ensure board copying creates a true deep copy without shared references."""
        self.board.place_stone(1, 1, BLACK)
        board_copy = self.board.copy()
        
        # Modify the copy
        board_copy.place_stone(2, 2, WHITE)
        
        # Verify the original is untouched
        self.assertEqual(self.board.grid[2][2], EMPTY)
        self.assertEqual(self.board.stone_count, 1)
        self.assertEqual(board_copy.stone_count, 2)

if __name__ == '__main__':
    unittest.main()