import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import game_utils as gu
from agents.agent_mcts import mcts as mcts
from agents.agent_mcts.tree import TreeNode


def test_many_simulation_runs():
    """
    Run many simulations alternating starting players to verify the
    simulation results are statistically balanced (no player advantage).
    The expected total_win_count should be close to zero within 3 standard deviations.
    """
    # switch players based on index to get "equal" winning chances overall (no advantage to starting player)
    total_win_count = 0

    N = 1000 # number of simulations
    for i in range(N):
        if (i%2 == 0): 
            player = gu.PLAYER1
            win_multiplier = 1
        else: 
            player = gu.PLAYER2            
            win_multiplier = -1
        board = gu.initialize_game_state()
        board_node = TreeNode(board, player=player)
        win_value, _ = mcts.simulation(board_node, player)
        total_win_count += win_value*win_multiplier

    # expected value: 0, accepted deviation: 3 * std (~99%)
    std = np.sqrt(N)

    assert np.abs(total_win_count) <= 3 * std, (
        "Total win count exceeds 3 standard deviations indicating possible bias or error in the simulation."
    )


def test_avoid_certain_defeat():
    """
    Test that the MCTS agent avoids a certain defeat.
    """
    board = gu.initialize_game_state()
    player = gu.PLAYER2
    col_idx = 3
    board[:, col_idx] = [1,1,1,0,0,0]
    action, _ = mcts.generate_move_mcts(board, player, saved_state=None)
    gu.apply_player_action(board, action, player)
    control_column = [1,1,1,2,0,0]
    assert np.all(board[:, col_idx] == control_column), (
        "Certain defeat not averted, possibly (but unprobable) due to chance."
    )


def test_achieve_certain_victory_vertical():
    """
    Test that the MCTS agent attains certain victory (vertically) 
    if presented with the chance to do so.
    """
    board = gu.initialize_game_state()
    player = gu.PLAYER1
    col_idx = 3
    board[:, col_idx] = [1,1,1,0,0,0]
    action, _ = mcts.generate_move_mcts(board, player, saved_state=None)
    gu.apply_player_action(board, action, player)
    assert gu.check_end_state(board, player, action), (
        "Certain victory (vertically) not attained."
    )


def test_achieve_certain_victory_horizontally():
    """
    Test that the MCTS agent attains certain victory (horizontally) 
    if presented with the chance to do so.
    """
    board = gu.initialize_game_state()
    player = gu.PLAYER1
    row_idx = 0
    board[row_idx, :] = [1,1,1,0,0,0,1]
    action, _ = mcts.generate_move_mcts(board, player, saved_state=None)
    gu.apply_player_action(board, action, player)
    assert gu.check_end_state(board, player, action), (
        "Certain victory (horizontally) not attained."
    )


def test_returns_unvisited_child_immediately():
    """
    Test that the UCT selection function prioritizes unvisited child.
    """
    board = gu.initialize_game_state()
    parent = TreeNode(board=board)
    child1 = TreeNode(board=board, parent=parent)
    child2 = TreeNode(board=board, parent=parent)
    # artificially set parameter of children nodes
    child1.visits = 5
    child1.wins = 3
    child2.visits = 0  # unvisited child, should be selected
    parent.children = [child1, child2]

    result = mcts.get_child_node_with_highest_UCT(parent)
    assert result == child2, (
        "Unvisited child was not selected."
    )


def test_returns_child_with_highest_uct_value():
    """
    Test that the UCT selection function correctly returns the child with the highest UCT score.
    """
    board = gu.initialize_game_state()
    parent = TreeNode(board=board)
    child1 = TreeNode(board=board, parent=parent)
    child2 = TreeNode(board=board, parent=parent)
    # both children visited
    child1.visits = 10
    child1.wins = 5
    child2.visits = 10
    child2.wins = 6
    parent.children = [child1, child2]
    parent.visits = 20

    result = mcts.get_child_node_with_highest_UCT(parent)
    # calculate UCT manually to verify expected best child
    exploit1 = child1.wins / child1.visits  # 0.5
    explore1 = np.sqrt(2) * np.sqrt(np.log(parent.visits) / child1.visits)
    uct1 = exploit1 + explore1

    exploit2 = child2.wins / child2.visits  # 0.6
    explore2 = np.sqrt(2) * np.sqrt(np.log(parent.visits) / child2.visits)
    uct2 = exploit2 + explore2

    expected = child1 if uct1 > uct2 else child2
    assert result is expected, (
        "Did not return child node with highest UCT score."
    )


def test_expansion_adds_child_to_parent():
    """
    Test that expansion adds a new child to the parent (root) node.
    """
    board = gu.initialize_game_state()
    root_node = TreeNode(board, player=gu.PLAYER1)
    child_node = mcts.expansion(root_node)
    assert child_node in root_node.children, (
        "Expansion did not add child to root node."
    )


def test_expansion_records_action_in_expanded_actions():
    """
    Test that expansion records the new action in the parent's expanded actions.
    """
    board = gu.initialize_game_state()
    root_node = TreeNode(board, player=gu.PLAYER1)
    child_node = mcts.expansion(root_node)
    assert child_node.previous_action in root_node.expanded_actions, (
        "Action is not recored in root's expanded actions."
    )


def test_expansion_child_player_is_opponent():
    """
    Test that expansion assigns the correct player (opponent) to the child node.
    """
    board = gu.initialize_game_state()
    root_node = TreeNode(board, player=gu.PLAYER1)
    child_node = mcts.expansion(root_node)
    expected_player = gu.BoardPiece(3 - root_node.player)
    assert child_node.player == expected_player, (
        "Incorrect player assigned to child node during expansion."
    )


def test_expansion_grand_child_player_is_org_player():
    """
    Test that expansion assigns the correct player (same as root player) to the grandchild node.
    """
    board = gu.initialize_game_state()
    root_node = TreeNode(board, player=gu.PLAYER1)
    child_node = mcts.expansion(root_node)
    grandchild_node = mcts.expansion(child_node) 
    expected_player = root_node.player
    assert grandchild_node.player == expected_player, (
        "Incorrect player assigned to grandchild node during expansion."
    )


def test_expansion_board_reflects_action():
    """
    Test that the child node's board accurately reflects the action taken during expansion.
    """
    board = gu.initialize_game_state()
    root_node = TreeNode(board, player=gu.PLAYER2)
    child_node = mcts.expansion(root_node)
    parent_board = root_node.board
    child_board = child_node.board
    action_col = child_node.previous_action
    col_diff = child_board[:, action_col] != parent_board[:, action_col]
    # exactly one element has changes
    assert np.sum(col_diff) == 1, (
        "Board of child node does not have the correct amount of changes compared to parent board"
    )


def test_simulation_win_value_is_valid():
    """
    Test that the simulation returns a valid win value: 1 (win), -1 (loss), or 0 (draw).
    """
    board = gu.initialize_game_state()
    node = TreeNode(board, player=gu.PLAYER1)
    win_value, _ = mcts.simulation(node)
    assert win_value in (-1, 0, 1), (
        "Win value should be one of -1, 0, or 1"
    )


def test_simulation_respects_max_simulation_depth():
    """
    Test that the simulation follows the imposed maximum depth limit.
    """
    board = gu.initialize_game_state()
    node = TreeNode(board, player=gu.PLAYER1)
    max_depth = 5
    _, move_count = mcts.simulation(node, max_simulation_depth=max_depth)
    assert move_count <= max_depth, (
        "Simulation move count should not exceed max_simulation_depth"
    )


def test_backpropagation_increments_visits_value_and_wins():
    """
    Test that backpropagation increments the visits, value, and wins for the root node.
    """
    board = gu.initialize_game_state()
    root = TreeNode(board=board)
    mcts.backpropagation(root, simulation_result=1)
    assert root.visits == root.value == root.wins == 1, (
        "Incorrect value results from backpropagation."
    )


def test_backpropagation_does_not_increment_wins_on_draw():
    """
    Test that backpropagation does not increment the win count when the simulation result is a draw.
    """
    board = gu.initialize_game_state()
    root = TreeNode(board=board)    
    mcts.backpropagation(root, simulation_result=0)
    assert root.wins == 0, (
        "Wins incremented even though no win occured."
    )


def test_backpropagation_flips_result_and_updates_parent_and_child():
    """
    Test that backpropagation flips the simulation result when moving up 
    the tree and updates the parent and child correctly.
    """
    board = gu.initialize_game_state()
    root = TreeNode(board=board)
    child = TreeNode(board=board, parent=root)
    mcts.backpropagation(child, simulation_result=1)
    assert (root.visits == 1 and root.value == -1 and root.wins == 0), (
        "Backpropagation did not correctly update parent node with simulation results."
    )


def test_get_all_valid_actions_returns_only_non_full_columns():
    """
    Test that get_all_valid_actions returns only columns that are not fully occupied.
    """
    board = gu.initialize_game_state()
    col_idx = gu.PlayerAction(0)
    board[:, col_idx] = gu.PLAYER1
    valid_actions = mcts.get_all_valid_actions(board)
    assert col_idx not in valid_actions, (
        "Index of full column not removed from valid actions."
    )