import numpy as np
from game_utils import BoardPiece, SavedState, PlayerAction, GameState
from game_utils import check_end_state, apply_player_action, get_lowest_empty_row, BOARD_COLS
from agents.agent_mcts.tree import TreeNode
from typing import Optional


def generate_move_mcts(board: np.ndarray, 
         player: BoardPiece, 
         saved_state: SavedState | None, 
         iterations=4000,
         max_depth = np.inf
         ) -> tuple[PlayerAction, SavedState]: 
    """
    Perform Monte Carlo Tree Search (MCTS) to determine the next action for the given board state.

    Parameters
    ----------
    board : np.ndarray
        The current board state.
    player : BoardPiece
        The player making the move.
    saved_state : SavedState or None
        A saved state from a previous call to MCTS, used to continue the search tree across turns.
    iterations : int, optional
        The number of MCTS iterations to perform. Default is 4000.
    max_depth : float, optional
        The maximum depth to explore in the tree. Default is np.inf (no depth limit).

    Returns
    -------
    tuple[PlayerAction, SavedState]
        A tuple containing:
        - The chosen action (PlayerAction) to play.
        - The updated saved state (SavedState) to carry over to the next turn.
    
    Notes
    -----
    The MCTS algorithm balances exploration and exploitation using the UCT score, 
    building a search tree to approximate the best action based on random simulations.
    """

    # player of root is the opponent
    prev_player = BoardPiece(1 + (2 - player)) 
    
    if saved_state: root = saved_state
    else: root = TreeNode(board, player=prev_player)
    num_visits = root.visits 

    for i in range(iterations-num_visits): # reduce number of iterations based on saved state visits
        selected_node = selection(root)
        expanded_node = expansion(selected_node)
        # also returns move count, currently not used
        simulation_results, _ = simulation(expanded_node, max_simulation_depth=max_depth)
        backpropagation(expanded_node, simulation_results)
    
    # always select child if it leads to certain victory
    for child in root.children:
        if check_end_state(child.board, child.previous_action, player) == GameState.IS_WIN:
            return child.previous_action, child
        
    # final selection based on number of visits
    best_child = max(root.children, key=lambda c: c.visits)
    
    return best_child.previous_action, best_child


def selection(node: TreeNode) -> TreeNode:
    """
    Select a node to be expanded in the Monte Carlo Tree Search (MCTS).

    The selection phase of MCTS works as follows:
    1) If the current node is not fully expanded (i.e., not all possible moves have been visited), 
       return it for expansion.
    2) If the node is fully expanded, select the child with the highest UCT score 
       and repeat until a non-fully-expanded node is found.

    Parameters
    ----------
    node : TreeNode
        The root node of the current MCTS search subtree.

    Returns
    -------
    TreeNode
        The selected node to expand further in the MCTS.
    """
    while True:
        all_valid_actions_count=len(get_all_valid_actions(node.board))
        # fully expand nodes, i.e. visit each (possible) child at least once
        if node.is_fully_expanded(all_valid_actions_count):
            if node.children:
                node = get_child_node_with_highest_UCT(node)
            else: return node
        else: return node


def get_child_node_with_highest_UCT(node: TreeNode, 
                                     explore_param=np.sqrt(2)) -> TreeNode:
    """
    Select the child node of the given node with the highest UCT (Upper Confidence Bound for Trees) score.

    The UCT score balances exploitation (average reward) and exploration (encouraging 
    visiting less-visited nodes). If a child node has not been visited, it is immediately 
    returned to ensure that every node is visited at least once.

    Parameters
    ----------
    node : TreeNode
        The current node whose children are to be evaluated.
    explore_param : float, optional
        The exploration parameter that controls the balance between exploration and 
        exploitation. Defaults to sqrt(2).

    Returns
    -------
    TreeNode
        The child node with the highest UCT score, or an unvisited child if available.
    """
    highest_uct_value = -np.inf # ensures that any uct value is larger 
    return_child = None
    for child in node.children:
        if child.visits == 0: # prioritize unvisited nodes
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
    """
    Expand the given node by creating a new (unexplored) child node.

    Parameters
    ----------
    node : TreeNode
        The node to be expanded.

    Returns
    -------
    TreeNode
        The newly created child node representing an unexplored action.
    """
    board = node.board.copy()
    # determine child player based on parent player
    child_player = BoardPiece(3 - node.player)   

    action = generate_random_move(
        board, 
        excluded_actions = node.expanded_actions # to get different actions/child nodes at each expansion
    )
    node.expanded_actions.append(action)
    apply_player_action(board, action, child_player)
    child = TreeNode(board, parent=node, player=child_player, previous_action=action) 
    node.add_child(child)
    return child


def simulation(node: TreeNode, max_simulation_depth=np.inf) -> tuple[int,int]:
    """
    Perform a random simulation from the given node until the game ends or a depth limit is reached.

    Parameters
    ----------
    node : TreeNode
        Starting node for the simulation.
        
    max_simulation_depth : float, optional
        Maximum number of moves to simulate before stopping. Defaults to np.inf,
        which means no depth limit (simulate until game ends).

    Returns
    -------
    win_value : int
        The outcome of the simulation from the starting player's perspective:
        1 for a win, -1 for a loss, and 0 for a draw.

    move_count : int
        The number of moves it took to reach the end of the game during the simulation.
    """
    
    starting_player = current_player = node.player
    board = node.board.copy()
    move_count = 0
    win_value = 0  # 0: draw, 1: win, -1: loss

    while move_count < max_simulation_depth:
        action = generate_random_move(board)
        apply_player_action(board, action, current_player)
        end_state = check_end_state(board, action, current_player)

        if end_state != GameState.STILL_PLAYING:
            if end_state == GameState.IS_DRAW:
                win_value = 0
            else:
                # determine win or loss from starting player's perspective
                win_value = 1 if current_player == starting_player else -1
            break
        # change player before next move
        current_player = BoardPiece(3-current_player)
        move_count += 1

    return win_value, move_count


def backpropagation(node: TreeNode, simulation_result: float) -> None:
    """
    Update the node and all its ancestors with the result of a simulation.

    Procedure:
    - Increment the visit count.
    - Update the value and win count of the node.
    - Alternate the perspective (switch players by flipping simulation result) as we traverse back up the tree.

    Parameters
    ----------
    node : TreeNode
        The node to start backpropagation from.
    
    simulation_result : float
        The simulation result from the perspective of the simulation-starting player:
        1 for a win, -1 for a loss, 0 for a draw.
    """
    while node:
        node.visits += 1
        node.value += simulation_result
        if simulation_result == 1:
            node.wins += 1  # increment wins only if player won
        node = node.parent
        simulation_result *= -1  # flip player perspective at each level


def generate_random_move(
    board: np.ndarray, 
    excluded_actions: list[PlayerAction] | None = None
) -> Optional[PlayerAction]:
    """
    Randomly select a valid action from the board that is not in the excluded actions.

    Parameters
    ----------
    board : np.ndarray
        The current game board state.
    excluded_actions : Sequence[PlayerAction], optional
        Actions to exclude from selection (default is empty list).

    Returns
    -------
    PlayerAction or None
        A randomly chosen valid action not in excluded_actions, or None if no valid actions remain.
    """
    if excluded_actions is None:
        excluded_actions = []

    all_valid_actions = get_all_valid_actions(board)
    # filter exluded actions using sets
    available_actions = list(set(all_valid_actions) - set(excluded_actions))

    if not available_actions:
        return None

    action = PlayerAction(np.random.choice(available_actions))
    return action


def get_all_valid_actions(board: np.ndarray) -> list[PlayerAction]:
    """
    Return all valid actions for the given board, i.e. columns that are not yet full.

    Parameters
    ----------
    board : np.ndarray
        The current game board.

    Returns
    -------
    List[PlayerAction]
        List of valid columns (=actions) where a piece can be placed.
    """
    all_valid_actions = []
    for col_i in range(BOARD_COLS):
        if get_lowest_empty_row(board, col_i) >= 0:
            all_valid_actions.append(col_i)
    return all_valid_actions