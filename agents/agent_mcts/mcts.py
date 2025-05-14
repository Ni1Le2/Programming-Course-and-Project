import numpy as np
from game_utils import *
import game_utils as gu
from agents.agent_mcts.tree import *
from tree import *

# https://www.youtube.com/watch?v=Fbs4lnGLS8M
# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
# https://www.youtube.com/watch?v=ghhznqBoESY

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

def mcts(board: np.ndarray, player: gu.BoardPiece, iterations = 1000) -> PlayerAction: 
    root = TreeNode(board, player=player)

    for i in range(iterations):
        selected_node = selection(root)
        expanded_node = expansion(selected_node)
        simulation_results = simulation(expanded_node)
        backpropagation(expanded_node, simulation_results)
        #player = gu.BoardPiece(1 - player)

    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action


def selection(node: TreeNode) -> TreeNode:
    '''
    Select node to be expanded based on if the node has been "fully extended" 
    (i.e. reached our desired depth?) or based on UCT score compared to sibling nodes.
    [First each child node will be investigated at least 1 (better: n?) times. 
    After that, a selection will be regarding which node to investiagte further.] 
    Final selection based on highest number of visits -> may require new function.
    '''
    while node.children:
        node = _get_child_node_with_highest_UCT(node)
    return node


def _get_child_node_with_highest_UCT(node: TreeNode, explore_param=np.sqrt(2)) -> TreeNode:
    '''
    Compute and compare all UCT values of children nodes and return child with largest value.
    '''
    highest_uct_value = 0
    for child in node.children:
        # avoid division by zero -> visit every node at least once
        if child.visits == 0: 
            return child
        exploitation_term = child.wins/child.visits
        exploration_term = explore_param*np.sqrt(np.log(node.visits)/child.visits)
        uct_value = exploitation_term + exploration_term
        if uct_value > highest_uct_value:
            highest_uct_value = uct_value
            return_child = child
    return return_child


def expansion(node: TreeNode) -> TreeNode:
    '''
    Add new node to the tree that will be investiagated.
    Returns updated tree with an additional (child) node? -> not necessary if we use this in a (recursive) loop?
    '''
    saved_state = {PLAYER1: None, PLAYER2: None}
    board = node.data
    # determine child player based on parent player (two player game)
    current_player = 1 - node.player

    action, saved_state[current_player] = generate_move_random(
        board.copy(),  # copy board to be safe, even though agents shouldn't modify it
        current_player, saved_state[current_player],
        excluded_actions = node.expanded_actions # to get different actions/child nodes at each expansion
    )
    node.expanded_actions.append(action)
    apply_player_action(board, action, current_player)
    child = TreeNode(board, parent=node, player=current_player) # board updated in place
    node.add_child(child)
    return child


def simulation(node:  np.ndarray, return_final_board=False):
    '''
    Randomly simulate moves on a given game board (node) until one player wins (leaf of tree).
    Returns who won (and maybe number of steps it took?): Win: 1, Loss: -1, Draw: 0.
    '''
    saved_state = {PLAYER1: None, PLAYER2: None}
    starting_player = current_player = node.player
    board = node.board
    simulating = True

    # it might be interesting to incorporate the move count into the win_value: 
    # are fast (random) losses "worse" than slow (random) losses? (somewhat of a heuristic)
    move_count = 0
    # result of the game from perspective of starting player: 0: draw, 1: win, -1: lost
    win_value = 0

    while simulating:
        action, saved_state[current_player] = generate_move_random(
            board.copy(),  # copy board to be safe, even though agents shouldn't modify it
            current_player, saved_state[current_player]
        )
        
        # should not be necessary for simulated games by agent, keep it for now to be safe
        move_status = check_move_status(board, action)
        if move_status != MoveStatus.IS_VALID:
            print(f'Move {action} is invalid: {move_status.value}')
            simulating = False
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
            simulating = False
            break
        # change player before next move
        current_player = gu.BoardPiece(1-current_player)
        move_count += 1
    
    if return_final_board: # for testing
        return win_value, move_count, board, action
    else: return win_value, move_count


def backpropagation(node: TreeNode, value: float):
    '''
    Update node and parents (up to root) with result of simulation.
    Simulation's result is backpropagated through the entire tree, updating all nodes.
    Win: 1, Loss: -1, Draw: 0
    Updates nodes values and increments the number of visits.
    Requires one step for each parent (or even node?) -> recursively
    '''
    while node:
        node.visits += 1
        node.value += value
        if value == 1:
            node.wins += 1  # increment wins only if player won
        node = node.parent


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None, excluded_actions=[]
) -> tuple[PlayerAction, SavedState | None]:
    # Choose a valid, non-full column randomly and return it as `action`
    all_valid_actions = _get_all_valid_actions(board)
    # take sets to easily filter exluded actions
    all_valid_actions = list(set(all_valid_actions) - set(excluded_actions))

    action_idx = np.random.randint(len(all_valid_actions))
    action = PlayerAction(all_valid_actions[action_idx])

    return action, saved_state


def _get_all_valid_actions(board: np.ndarray) -> PlayerAction:
    all_valid_actions = []
    for col_i in range(BOARD_COLS):
        if gu.get_lowest_empty_row(board, col_i) >= 0:
            all_valid_actions.append(col_i)
    return all_valid_actions