# from https://medium.com/pythoneers/getting-started-with-trees-in-python-a-beginners-guide-4e68818e7c05
import numpy as np
import game_utils as gu
import agents.agent_mcts.mcts as mcts
from typing import Optional

class TreeNode:
    def __init__(self, board: np.ndarray, 
                previous_action: Optional[gu.PlayerAction]=None,
                parent: Optional["TreeNode"]=None, 
                player: Optional[gu.BoardPiece]=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.value = 0 # value of node, changed through simulation and backpropagation
        self.wins = 0
        self.visits = 0 # how often node has been visited
        self.uct = 0
        self.player = player
        self.previous_action = previous_action
        self.expanded_actions = []

    def add_child(self, child):
        self.children.append(child)

    def is_fully_expanded(self):
        # make nicer later
        return len(self.expanded_actions) >= len(mcts._get_all_valid_actions(self.board))


# Create a function for insertion
def insert_node(root, node):
    if root is None:
        root = node
    else:
        root.add_child(node)


# Create a function for deletion
def delete_node(root, target):
    if root is None:
        return None
    root.children = [child for child in root.children if child.data != target]
    for child in root.children:
        delete_node(child, target)


# Create a function to calculate the height of a tree
def tree_height(node):
    if node is None:
        return 0
    if not node.children:
        return 1
    return 1 + max(tree_height(child) for child in node.children)
