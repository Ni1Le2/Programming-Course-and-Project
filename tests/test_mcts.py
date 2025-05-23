import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agent_mcts.mcts import simulation, mcts
from game_utils import initialize_game_state, pretty_print_board, PLAYER1, PLAYER2, PLAYER1_PRINT
import game_utils as gu

def test_many_simulation_runs():
    # switch players based on index to get "equal" winning chances overall (no advantage to starting player)
    total_win_count = 0

    N = 1000 # number of simulations, 10000 takes a long time, 1000 is fine
    for i in range(N):
        if (i%2 == 0): 
            player = PLAYER1
            win_multiplier = 1
        else: 
            player = PLAYER2            
            win_multiplier = -1
        board = initialize_game_state()
        win_value, move_count = simulation(board, player)
        total_win_count += win_value*win_multiplier
    print(total_win_count)
    # we want to be withing 3*stdv of the expected value (=0) which contains >99% of possible values
    # we can still be outside this range by chance, but it is very unlikely!
    std = np.sqrt(N)
    assert np.abs(total_win_count) <= 3*std


def test_avoid_certain_defeat():
    board = initialize_game_state()
    player = PLAYER2
    col_idx = 3
    board[:, col_idx] = [1,1,1,0,0,0]
    action, _ = mcts(board, player, saved_state=None)
    gu.apply_player_action(board, action, player)
    control_column = [1,1,1,2,0,0]
    assert np.all(board[:, col_idx] == control_column), "certain defeat not averted, possibly (but unprobable) due to chance"


def test_achieve_certain_victory_vertical():
    board = initialize_game_state()
    player = PLAYER1
    col_idx = 3
    board[:, col_idx] = [1,1,1,0,0,0]
    action, _ = mcts(board, player, saved_state=None)
    gu.apply_player_action(board, action, player)
    assert gu.check_end_state(board, player, action), "certain victory (vertical) not attained"


def test_achieve_certain_victory_horizontally():
    board = initialize_game_state()
    player = PLAYER1
    row_idx = 0
    board[row_idx, :] = [1,1,1,0,0,0,1]
    action, _ = mcts(board, player, saved_state=None)
    gu.apply_player_action(board, action, player)
    assert gu.check_end_state(board, player, action), "certain victory (horizontal) not attained"
