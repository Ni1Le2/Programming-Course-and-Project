import numpy as np
import sys
import os

# add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import game_utils as gu


def test_initial_game_board_display():
    """Test initialization of board."""
    board = gu.initialize_game_state()
    init_board = gu.pretty_print_board(board)
    expected = (
        "|==============|\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )
    assert init_board == expected, (
        "Initial board display does not match expected output.")
    

def test_random_board_to_string_and_back_to_matrix():
    """Test reconstruction (transformation to string and then back to np.ndarray) of random board."""
    random_board = gu.create_random_game_state()
    # transform board to string and then back to np.ndarray()
    random_pretty_board = gu.pretty_print_board(random_board)
    reconstructed_board = gu.string_to_board(random_pretty_board)
    assert np.all(random_board == reconstructed_board), ( 
        "Board reconstruction is not identical to original board.")


def test_lowest_empty_row_of_empty_board():
    """Test lowest empty row detection for an empty column."""
    # intialize empty board
    board = gu.initialize_game_state()
    # set col_idx and create predefined column
    col_idx = 2
    lowest_row = gu.get_lowest_empty_row(board, col_idx)
    true_lowest_row = 0
    assert lowest_row == true_lowest_row, ( 
        "Lowest row of empty not identical to expected value.")
    

def test_finding_lowest_empty_row_in_column():
    """Test lowest empty row detection for a partially filled column."""
    # intialize empty board
    board = gu.initialize_game_state()
    # set col_idx and create predefined column
    col_idx = 2
    board[:, col_idx] = np.array([1,2,1,0,0,0])
    lowest_row = gu.get_lowest_empty_row(board, col_idx)
    true_lowest_row = 3
    assert lowest_row == true_lowest_row, ( 
        "Lowest row not identical to expected value.")


def test_lowest_empty_row_of_full_column():
    """Test that the lowest empty row returns -1 for a full column."""
    board = gu.initialize_game_state()
    # set col_idx and create predefined column
    col_idx = 2
    board[:, col_idx] = np.array([1,2,1,2,1,2])
    lowest_row = gu.get_lowest_empty_row(board, 2)
    true_lowest_row = -1
    assert lowest_row == true_lowest_row, ( 
        "Lowest row of full column not identical to expected value (-1).")


def test_making_a_single_move():
    """Test applying a single player move to an empty column."""
    board = gu.initialize_game_state()
    col_idx = 3
    gu.apply_player_action(board, col_idx, player=gu.PLAYER1)
    output_column = board[:,col_idx]
    control_column = np.array([1,0,0,0,0,0])
    assert np.all(output_column == control_column), ( 
        "Action (single move) did not result in expected column.")


def test_making_multiple_moves_in_one_column():
    """Test applying multiple player moves to the same column."""
    board = gu.initialize_game_state()
    col_idx = 3
    gu.apply_player_action(board, col_idx, player=gu.PLAYER1)
    gu.apply_player_action(board, col_idx, player=gu.PLAYER2)
    gu.apply_player_action(board, col_idx, player=gu.PLAYER1)
    output_column = board[:,col_idx]
    control_column = np.array([1,2,1,0,0,0])
    assert np.all(output_column == control_column), ( 
        "Action (multiple moves) did not result in expected column.")


def test_detecting_four_connected_pieces():
    """
    Test that four_connected_pieces correctly detects four consecutive pieces
    for a given player in a 1D numpy array representing a row or column.
    """
    bool_res = []
    
    player_1_win = np.array([0,2,1,1,1,1,0])
    bool_res.append(gu.four_connected_pieces(player_1_win, gu.PLAYER1))
    
    player_2_win = np.array([0,0,0,2,2,2,2])
    bool_res.append(gu.four_connected_pieces(player_2_win, gu.PLAYER2))
    
    no_player_wins = np.array([0,1,1,1,0,1,0])
    bool_res.append(gu.four_connected_pieces(no_player_wins, gu.PLAYER1))

    control_bool = [True, True, False]
    
    assert bool_res == control_bool, (
        "At least one horizontal check of four connected pieces does not match expected outcomes."
    )


def test_detecting_four_connected_pieces_on_diagonal():
    """
    Test detection of four connected pieces along the main diagonal of the board.
    """
    board = gu.initialize_game_state()
    for idx in range(gu.BOARD_ROWS):
        board[idx,idx] = 1
    diagonal, anti_diagonal = gu.extract_diagonals(board, row_idx=0, col_idx=0)
    assert gu.four_connected_pieces(diagonal, gu.PLAYER1), (
        "Win across diagonal was not detected."
    )


def test_detecting_four_connected_pieces_on_anti_diagonal():
    """
    Test detection of four connected pieces along the anti-diagonal of the board.
    """
    board = gu.initialize_game_state()
    for idx in range(gu.BOARD_ROWS):
        board[idx,gu.BOARD_COLS-1-idx] = 1
    diagonal, anti_diagonal = gu.extract_diagonals(board, row_idx=0, col_idx=gu.BOARD_COLS - 1)
    assert gu.four_connected_pieces(anti_diagonal, player=gu.PLAYER1), (
        "Win across anti-diagonal was not detected."
    )


def test_if_game_board_is_full():
    """Tests whether randomly created full board is detected as full."""
    board = gu.create_random_game_state(full_board=True)
    assert gu.is_full(board), (
        "Board not detected as full."
    )


def test_full_column_status():
    """Test that check_move_status returns FULL_COLUMN when the specified column is completely filled."""
    board = gu.create_random_game_state(full_board=True)
    col_idx = gu.PlayerAction(2)
    assert gu.check_move_status(board, col_idx) == gu.MoveStatus.FULL_COLUMN, (
        "MoveStatus for column in full random board is not FULL_COLUMN."
    )


def test_out_of_bounds():
    """Test that check_move_status returns OUT_OF_BOUNDS when the column index is outside valid range."""
    board = gu.initialize_game_state()
    col_idx = gu.PlayerAction(gu.BOARD_COLS)
    assert gu.check_move_status(board, col_idx) == gu.MoveStatus.OUT_OF_BOUNDS, (
        "MoveStatus for column index out of bounds is not OUT_OF_BOUNDS."
    )


def test_wrong_type():
    """Test that check_move_status returns WRONG_TYPE when the column argument is of incorrect type."""
    board = gu.initialize_game_state()
    col_idx = str(1)
    assert gu.check_move_status(board, col_idx) == gu.MoveStatus.WRONG_TYPE, (
        "MoveStatus for incorrect column data type is note WRONG_TYPE."
    )
