import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, BOARD_COLS, get_lowest_empty_row

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    # choose a valid, non-full column randomly and return it as `action`
    valid_action = False
    # never finds valid action if game is full, but in that case we have a draw
    while not valid_action:
        action = np.random.randint(BOARD_COLS)
        if get_lowest_empty_row(board, action) >= 0:
            valid_action = True
    action = PlayerAction(action)
    return action, saved_state
    