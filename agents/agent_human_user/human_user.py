import numpy as np
from typing import Callable, Any

from game_utils import BoardPiece, PlayerAction, SavedState, MoveStatus, check_move_status


def query_user(prompt_function: Callable) -> Any:
    usr_input = prompt_function("Column? ")
    return usr_input


def user_move(board: np.ndarray,
              player: BoardPiece,
              saved_state: SavedState | None) -> tuple[PlayerAction, SavedState | None]:
    """
    Handles the user's move input by repeatedly prompting until a valid move is provided.

    Parameters
    ----------
    board : np.ndarray
        The current game board.

    player : BoardPiece
        The current player making a move.

    saved_state : SavedState | None
        Any saved state relevant to the current player’s turn.

    Returns
    -------
    tuple[PlayerAction, SavedState | None]
        The valid player action and the updated saved state.
    """
    move_status = None
    while move_status != MoveStatus.IS_VALID:
        input_move_string = query_user(input)
        input_move = convert_str_to_action(input_move_string)
        if input_move is None:
            continue
        move_status = check_move_status(board, input_move)
        if move_status != MoveStatus.IS_VALID:
            print(f'Move is invalid: {move_status.value}')
            print('Try again.')
    return input_move, saved_state


def convert_str_to_action(input_move_string: str) -> PlayerAction | None:
    """
    Convert a string input from the user to a PlayerAction (int). If conversion fails,
    inform user and return None.
    """
    try:
        input_move = PlayerAction(input_move_string)
    except ValueError:
        input_move = None
        print('Invalid move: Input must be an integer.')
        print('Try again.')
    return input_move