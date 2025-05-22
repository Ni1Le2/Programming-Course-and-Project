import numpy as np
from game_utils import *
import game_utils as gu
from agents.agent_mcts.tree import *

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

def mcts(board: np.ndarray, player: gu.BoardPiece, 
         saved_state: SavedState | None, iterations=1500) -> PlayerAction: 
    """
    
    """
    # IMPLEMENT IF SAVED_STATE:
    if saved_state: pass
    root = TreeNode(board, player=player)
    for i in range(iterations):
        selected_node = selection(root)
        expanded_node = expansion(selected_node)
        # also returns move count, maybe useful later
        simulation_results, _ = simulation(expanded_node)
        backpropagation(expanded_node, simulation_results)
    
    # Final selection based on highest number of visits
    best_child = max(root.children, key=lambda c: c.visits)
    
    best_action = best_child.previous_action
    # save the relevant part of the tree
    saved_state = best_child

    return best_action, saved_state


def selection(node: TreeNode) -> TreeNode:
    '''
    Select node to be expanded based on 1) whether the node has been "fully extended" 
    (i.e. all valid moves have been visited at least once) or 2) based on UCT score 
    of children nodes. Firstly, each child node will be investigated at least once.
    After that, a selection will be made regarding which node to investiagte further,
    considering both exploration and exploitation. 
    '''
    while True:
        # fully expand nodes, i.e. visit each child at least once
        if node.is_fully_expanded():
            if node.children:
                node = _get_child_node_with_highest_UCT(node)
            else: return node
        else: return node


def _get_child_node_with_highest_UCT(node: TreeNode, 
                                     explore_param=np.sqrt(2)) -> TreeNode:
    '''
    Iterate through all children and return child with highest UCT score.
    If child has not been visited yet, it is selected (favoring exploration)
    ensuring that every node is visited at least once.
    '''
    highest_uct_value = -np.inf # to ensure that any uct value is larger 
    return_child = None
    for child in node.children:
        # avoids division by zero, ensures visiting every node at least once
        if child.visits == 0: 
            return child
        else:
            exploitation_term = child.wins/child.visits
            exploration_term = explore_param*np.sqrt(np.log(node.visits)/child.visits)
            uct_value = exploitation_term + exploration_term
            if uct_value > highest_uct_value:
                highest_uct_value = uct_value
                return_child = child
    return return_child


def expansion(node: TreeNode) -> TreeNode:
    '''
    Create new unexplored child node and returns it. 
    '''
    saved_state = {PLAYER1: None, PLAYER2: None}
    board = node.board.copy()
    # determine child player based on parent player
    # two player game, Player1 = 1, Player2 = 2 
    child_player = BoardPiece(1 + (2 - node.player))    

    action = generate_random_move(
        board, 
        excluded_actions = node.expanded_actions # to get different actions/child nodes at each expansion
    )
    node.expanded_actions.append(action)
    apply_player_action(board, action, child_player)
    child = TreeNode(board, parent=node, player=child_player, previous_action=action) 
    node.add_child(child)
    return child


def simulation(node:  np.ndarray, return_final_board=False, max_simulation_depth=np.inf):
    '''
    Randomly simulate moves on a given game board (node) until game ends.
    Returns who won (and maybe number of steps it took?): Win: 1, Loss: -1, Draw: 0.

    Parameters
    ----------
    max_simulation_depth: float, default: np.inf
        Controls maximum amount of future steps in simulation (depth).
        Can be set to avoid long (mostly drawn out) games.
    '''

    saved_state = {PLAYER1: None, PLAYER2: None}
    starting_player = current_player = node.player
    board = node.board.copy()
    simulating = True

    # it might be interesting to incorporate the move count into the win_value: 
    # are fast (random) losses "worse" than slow (random) losses? (somewhat of a heuristic)
    move_count = 0

    # result of the game from perspective of starting player: 0: draw, 1: win, -1: lost
    win_value = 0

    while simulating and move_count < max_simulation_depth:
        action = generate_random_move(
            board 
        )
        
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
        current_player = gu.BoardPiece(2+(1-current_player))
        move_count += 1
    
    if return_final_board: # for testing
        return win_value, move_count, board, action
    else: return win_value, move_count


def backpropagation(node: TreeNode, value: float):
    '''
    Update node and all its parents (up to root) with result of simulation:
    - increase number of visits by 1
    - add simulation result value (Win: 1, Loss: -1, Draw: 0)
    - add win value if simulation was won
    '''
    while node:
        node.visits += 1
        node.value += value
        if value == 1:
            node.wins += 1  # increment wins only if player won
        value *= -1  # flip player perspective at each level
        node = node.parent


# no need to save states or pass player as these are just randomly generated moves
def generate_random_move(
    board: np.ndarray, 
    excluded_actions=[]) -> tuple[PlayerAction, SavedState | None]:
    """ 
    Choose a valid (not full) and not yet explored action randomly and 
    return it. If no valid action are available, return None.
    """

    all_valid_actions = _get_all_valid_actions(board)
    # take sets to easily filter exluded actions
    available_actions = list(set(all_valid_actions) - set(excluded_actions))

    if not available_actions:
        return None

    action = PlayerAction(np.random.choice(available_actions))
    return action


def _get_all_valid_actions(board: np.ndarray) -> PlayerAction:
    """
    Returns all valid actions for the given board, i.e. columns that are not yet full.
    """
    all_valid_actions = []
    for col_i in range(BOARD_COLS):
        if gu.get_lowest_empty_row(board, col_i) >= 0:
            all_valid_actions.append(col_i)
    return all_valid_actions