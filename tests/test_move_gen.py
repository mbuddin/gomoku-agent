import unittest
from unittest.mock import Mock

# Adjust these imports based on your exact project structure
from gomoku.move_gen import get_candidate_moves
from gomoku.board import EMPTY

class TestMoveGen(unittest.TestCase):

    def setUp(self):
        """Set up a mock board for testing before each test case."""
        self.board = Mock()
        self.board.size = 15
        # Initialize an empty 15x15 grid
        self.board.grid = [[EMPTY for _ in range(15)] for _ in range(15)]
        
        # Mock the get_empty_cells fallback
        self.board.get_empty_cells.return_value = []

        # Mock is_valid_move to return True if the cell is within bounds and EMPTY
        def mock_is_valid_move(row, col):
            if 0 <= row < 15 and 0 <= col < 15:
                return self.board.grid[row][col] == EMPTY
            return False
        self.board.is_valid_move.side_effect = mock_is_valid_move

    def test_empty_board_returns_center(self):
        """If the board is completely empty, it should return exactly the center."""
        moves = get_candidate_moves(self.board, radius=2)
        
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0], (7, 7))

    def test_single_stone_radius_1(self):
        """Test candidate generation around a single stone with radius 1."""
        # Place a stone at (7, 7)
        self.board.grid[7][7] = 1  # 1 represents a black stone, for example
        
        moves = get_candidate_moves(self.board, radius=1)
        
        # A radius of 1 should generate a 3x3 grid around the stone, minus the stone itself = 8 moves
        self.assertEqual(len(moves), 8)
        self.assertIn((6, 6), moves)
        self.assertIn((8, 8), moves)
        self.assertNotIn((7, 7), moves) # The occupied cell shouldn't be a candidate

    def test_overlapping_radii(self):
        """Ensure that two stones close to each other don't generate duplicate candidate moves."""
        self.board.grid[7][7] = 1
        self.board.grid[7][8] = 2
        
        moves = get_candidate_moves(self.board, radius=1)
        
        # 3x4 bounding box = 12 cells. Minus 2 occupied cells = 10 unique candidates.
        self.assertEqual(len(moves), 10)
        # Convert to a set to verify no duplicates exist
        self.assertEqual(len(moves), len(set(moves)))

    def test_board_edges(self):
        """Ensure the generator respects board boundaries using the is_valid_move mock."""
        # Place a stone in the absolute top-left corner
        self.board.grid[0][0] = 1
        
        moves = get_candidate_moves(self.board, radius=1)
        
        # Should only generate 3 valid adjacent moves: (0,1), (1,0), (1,1)
        self.assertEqual(len(moves), 3)
        self.assertIn((0, 1), moves)
        self.assertIn((1, 0), moves)
        self.assertIn((1, 1), moves)
        # Should not generate negative indices like (-1, -1)
        for r, c in moves:
            self.assertTrue(r >= 0 and c >= 0)

    def test_sorting_distance_to_center(self):
        """Ensure candidates are sorted closest to the center first."""
        # Place a stone somewhat off-center
        self.board.grid[5][5] = 1
        
        moves = get_candidate_moves(self.board, radius=1)
        
        # Center is (7, 7). The closest candidate to (7,7) among radius 1 from (5,5) is (6,6)
        # Distance of (6,6) to center is |6-7| + |6-7| = 2.
        self.assertEqual(moves[0], (6, 6))

if __name__ == '__main__':
    unittest.main()