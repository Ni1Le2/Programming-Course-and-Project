import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agent_mcts.mcts import simulation
from game_utils import initialize_game_state, pretty_print_board, PLAYER1, PLAYER2, PLAYER1_PRINT

# this is not a "real/correct" test funciton, but it helped me check whether th
def test_single_simulation_run():
    board = initialize_game_state()
    win_value, move_count, result_board, last_action = simulation(board, PLAYER1, return_final_board=True)
    print("starting player symbol: " + PLAYER1_PRINT)
    print("results from perspective of starting player: " + str(win_value))
    print(last_action)
    print(pretty_print_board(result_board))
    # assert False # assert False to see/check printed outputs
    assert True # not a real test!


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
