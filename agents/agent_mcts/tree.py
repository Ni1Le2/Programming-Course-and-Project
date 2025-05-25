"""
TreeNode class inspired by:
https://medium.com/pythoneers/getting-started-with-trees-in-python-a-beginners-guide-4e68818e7c05
"""

import numpy as np
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from game_utils import PlayerAction, BoardPiece

class TreeNode:
    def __init__(self, 
                 board: np.ndarray,
                 previous_action: Optional["PlayerAction"] = None,
                 parent: Optional["TreeNode"] = None,
                 player: Optional["BoardPiece"] = None):
        self.board: np.ndarray = board
        self.previous_action: Optional["PlayerAction"] = previous_action
        self.parent: Optional["TreeNode"] = parent
        self.player: Optional["BoardPiece"] = player

        self.children: List["TreeNode"] = []
        self.value: int = 0.0
        self.wins: int = 0
        self.visits: int = 0
        self.uct_score: float = 0.0
        self.expanded_actions: List["PlayerAction"] = []

    def add_child(self, child: "TreeNode") -> None:
        """Adds a child node to the current node."""
        self.children.append(child)

    def is_fully_expanded(self, all_valid_actions_count: int) -> bool:
        """Returns True if this node has expanded all valid actions."""
        return len(self.expanded_actions) >= all_valid_actions_count
