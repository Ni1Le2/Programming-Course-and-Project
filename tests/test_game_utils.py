import numpy as np
# should later be in agents.game_utils?
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import game_utils as gu
from game_utils import pretty_print_board, initialize_game_state, string_to_board, BOARD_SHAPE

def test_initial_game_board_display():
    board = initialize_game_state()
    init_board = pretty_print_board(board)
    # this is hard coded and ugly...
    board_control_string = "|==============|\n|              |\n|              |\n|              |\n|              |\n|              |\n|              |\n|==============|\n|0 1 2 3 4 5 6 |"
    assert init_board == board_control_string, 'board does not look like expected; have # of rows/cols changed?!'
    
def test_random_game_board_shape():
    board = initialize_game_state()
    init_board = pretty_print_board(board)
    random_board = gu.create_random_game_state()
    random_pretty_board = pretty_print_board(random_board)
    assert len(random_pretty_board) == len(init_board), "string lengths do not match"

def test_random_board_to_string_and_back_to_matrix():
    random_board = gu.create_random_game_state()
    random_pretty_board = pretty_print_board(random_board)
    reconstructed_board = string_to_board(random_pretty_board)
    assert np.all(random_board == reconstructed_board), "board reconstruction is not identical to original board"

def test_finding_lowest_empty_row_in_column():
    from game_utils import get_lowest_empty_row
    # create correctly shaped board
    board = gu.create_random_game_state()
    # set col_idx and create predefined column
    col_idx = 2
    board[:, col_idx] = [1,2,1,0,0,0]
    lowest_row = get_lowest_empty_row(board, 2)
    true_lowest_row = 3
    assert lowest_row == true_lowest_row

def test_lowest_empty_row_of_full_column():
    board = gu.create_random_game_state()
    # set col_idx and create predefined column
    col_idx = 2
    board[:, col_idx] = [1,2,1,2,1,2]
    lowest_row = gu.get_lowest_empty_row(board, 2)
    print(lowest_row)
    assert not lowest_row

def test_making_a_single_first_move():
    board = initialize_game_state()
    col_idx = 3
    gu.apply_player_action(board, col_idx, player=gu.PLAYER1)
    output_column = board[:,col_idx]
    control_column = [1,0,0,0,0,0]
    assert np.all(output_column == control_column)

def test_making_multipl_moves_in_one_column():
    board = initialize_game_state()
    col_idx = 3
    gu.apply_player_action(board, col_idx, player=gu.PLAYER1)
    gu.apply_player_action(board, col_idx, player=gu.PLAYER2)
    gu.apply_player_action(board, col_idx, player=gu.PLAYER1)
    output_column = board[:,col_idx]
    control_column = [1,2,1,0,0,0]
    assert np.all(output_column == control_column)

def test_horizontal_win_check():
    board = gu.initialize_game_state()
    # set row_idx and create predefined row
    row_idx = 2
    board[row_idx, :] = [0,2,1,1,1,1,0]
    bool_res = []
    bool_res.append(gu.horizontal_win_check(board[row_idx], gu.PLAYER1))
    board[row_idx, :] = [0,0,0,2,2,2,2]
    bool_res.append(gu.horizontal_win_check(board[row_idx], gu.PLAYER2))
    board[row_idx, :] = [0,1,1,1,0,1,0]
    bool_res.append(gu.horizontal_win_check(board[row_idx], gu.PLAYER1))
    control_bool = [True, True, False]
    assert bool_res == control_bool

def test_vertical_win_check():
    board = gu.initialize_game_state()
    # set row_idx and create predefined row
    col_idx = 2
    board[:, col_idx] = [1,1,1,1,1,1]
    bool_res = []
    bool_res.append(gu.vertical_win_check(board[:,col_idx], gu.PLAYER1))
    board[:, col_idx] = [0,0,2,2,2,2]
    bool_res.append(gu.vertical_win_check(board[:,col_idx], gu.PLAYER2))
    board[:, col_idx] = [0,1,1,1,0,1]
    bool_res.append(gu.vertical_win_check(board[:,col_idx], gu.PLAYER1))
    control_bool = [True, True, False]
    assert bool_res == control_bool

def test_diagonal_win_check():
    board = gu.initialize_game_state()
    for idx in range(gu.BOARD_ROWS):
        board[idx,idx] = 1
    assert gu.diagonal_win_check(board, 2, 2, gu.PLAYER1)

def test_if_game_is_drawn():
    # creates a nice 1-2-1-2 board, the perfect draw :)
    board = gu.initialize_game_state()
    for i in range(board.shape[0]):
        board[i,i%2::2] = 1
        board[i,(i+1)%2::2] = 2
    assert gu.is_draw(board)

def test_full_column_status():
    board = gu.initialize_game_state()
    board[:,:] = 1
    assert gu.check_move_status(board, column=np.int8(2)) == gu.MoveStatus.FULL_COLUMN

def test_out_of_bounds():
    board = gu.initialize_game_state()
    assert gu.check_move_status(board, column=np.int8(gu.BOARD_COLS+1)) == gu.MoveStatus.OUT_OF_BOUNDS

def test_wrong_type():
    board = gu.initialize_game_state()
    assert gu.check_move_status(board, column=[1,2]) == gu.MoveStatus.WRONG_TYPE
