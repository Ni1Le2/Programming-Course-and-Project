from typing import Callable
import time
from game_utils import PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT, GameState, MoveStatus, GenMove, BoardPiece
from game_utils import initialize_game_state, pretty_print_board, apply_player_action, update_saved_state, check_end_state, check_move_status
from agents.agent_human_user import user_move
from agents.agent_random import generate_move_random
from agents.agent_mcts import generate_move_mcts
import numpy as np

def play(
    mode = None,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):
    """
    """
    if mode == None:
        mode = int(input("Select mode:  \n 0 = player vs. player \n 1 = player vs. agent \n 2 = agent vs. agent \n 3 = agent vs. random agent \n"))
    
    if mode == 0: # player vs. player
        generate_move_1: GenMove = user_move
        generate_move_2: GenMove = user_move
    elif mode == 1: # player vs. agent
        generate_move_1: GenMove = user_move
        generate_move_2: GenMove = generate_move_mcts
    elif mode == 2: # agent vs. agent
        generate_move_1: GenMove = generate_move_mcts
        generate_move_2: GenMove = generate_move_mcts
    elif mode == 3: # agent vs. random_agent
        generate_move_1: GenMove = generate_move_random
        generate_move_2: GenMove = generate_move_mcts
    else:
        raise ValueError("Incorret mode selected. Please select valid mode (0, 1, 2, or 3)")
        
    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        action = None

        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                )
                
                # idea: 
                # saved_state of each player is the board (as a node with all children) 
                # after the last move they made.
                # Opponent makes move (action) based on which we update the saved state 
                # (i.e. select appropriate child or update board)

                if saved_state[player]: # does this work for None?
                    saved_state[player] = update_saved_state(saved_state[player], action)

                action, saved_state[player] = gen_move(
                    board.copy(),  # copy board to be safe, even though agents shouldn't modify it
                    player, 
                    saved_state[player], 
                    *args
                )

                print(f'Move time: {time.time() - t0:.3f}s')

                move_status = check_move_status(board, action)
                if move_status != MoveStatus.IS_VALID:
                    print(f'Move {action} is invalid: {move_status.value}')
                    print(f'{player_name} lost by making an illegal move.')
                    playing = False
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, action, player)

                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print('Game ended in draw')
                    else:
                        print(
                            f'{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                        )
                    playing = False
                    break


if __name__ == "__main__":
    play()


