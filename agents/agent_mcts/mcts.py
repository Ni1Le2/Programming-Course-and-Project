import numpy as np
from game_utils import *
import game_utils as gu
from agents.agent_mcts.tree import *

# https://www.youtube.com/watch?v=Fbs4lnGLS8M
# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/

# keep in mind:
# - root is the current game state
# - first level children will be opponents moves (and we count opponents wins)
# - second level will be "our" move, and so on
# - selection: maximize wins of our agent or minimize opponents wins, depending on level
# - fixed number of levels (let's start with 3 or 4) and iterations (? -> 1000 should be reasonable)

# general procedure:
# start with root (=board) -> expand child 1 -> simulate -> update -> expand child 2 -> ... 
# -> select most "promising" child -> expand "grandchild" 1 -> simulate -> update (both grandchild and most promising child) -> expand "grandhcild" 2 ->...
# until depth is reached? then reselect from first level (if there is one with better "potential", i.e. random win probability)

# notes:
# - it seems reasonable o do multiple (maybe 5 or 10?) initial simulations for each level 1 child
# - for lower levels we can probably decrease the number of simulations per expansion 
# - we want to store the tree so that we can reuse the relevant branches that the game actually followed

def expansion(board: np.ndarray, player: BoardPiece, action: PlayerAction):
    '''
    Add new node to the tree that will be investiagated.
    Returns updated tree with an additional (child) node.
    '''
    saved_state = {PLAYER1: None, PLAYER2: None}
    root = TreeNode(board)
    action, saved_state[player] = generate_move_random(
        board.copy(),  # copy board to be safe, even though agents shouldn't modify it
        player, saved_state[player]
    )
    apply_player_action(board, action, player)
    child = TreeNode(board)
    root.add_child(child)


def simulation(board:  np.ndarray, starting_player: BoardPiece, return_final_board=False):
    '''
    Randomly simulate moves on a given game board (node) until one player wins (leaf of tree).
    Returns who won (and maybe number of steps it took?).
    '''
    saved_state = {PLAYER1: None, PLAYER2: None}

    # determine current player and opponent
    if starting_player == PLAYER1: opponent = PLAYER2
    else: opponent = PLAYER1
    players = np.array([starting_player, opponent])

    playing = True
    # move count to determine who's turn it is and who won; even or uneven
    move_count = 0
    # result of the game from perspective of starting player: 0: draw, 1: win, -1: lost
    win_value = 0
    while playing:
        # get current player based on move count
        current_player = players[move_count%2]
        action, saved_state[current_player] = generate_move_random(
            board.copy(),  # copy board to be safe, even though agents shouldn't modify it
            current_player, saved_state[current_player]
        )
        
        # should not be necessary for simulated games by agent, keep it for now to be safe
        move_status = check_move_status(board, action)
        if move_status != MoveStatus.IS_VALID:
            print(f'Move {action} is invalid: {move_status.value}')
            playing = False
            break

        apply_player_action(board, action, current_player)
        end_state = check_end_state(board, action, current_player)

        if end_state != GameState.STILL_PLAYING:
            if end_state == GameState.IS_DRAW:
                win_value = 0
            else:
                if current_player == starting_player:
                    win_value = 1
                else: win_value = -1
            playing = False
            break
        move_count += 1
    # it might be interesting to incorporate the move count into the win_value: 
    # are fast (random) losses "worse" than slow (random) losses (somewhat of a heuristic)
    if return_final_board:
        return win_value, move_count, board, action
    else: return win_value, move_count

def selection():
    '''
    First each child node will be investigated at least 1 (better: n?) times. 
    After that, a selection will be regarding which node to investiagte further. 
    '''
    pass


def update():
    '''
    Update node and parents with result of simulation.
    '''
    pass


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    # Choose a valid, non-full column randomly and return it as `action`
    valid_action = False
    # never finds valid action if game is full, but in that case we have a draw
    while not valid_action:
        action = np.random.randint(BOARD_COLS)
        if get_lowest_empty_row(board, action) >= 0:
            valid_action = True
    action = PlayerAction(action)
    return action, saved_state