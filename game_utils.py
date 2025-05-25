from typing import Callable, Optional, TYPE_CHECKING
from enum import Enum
import numpy as np

# avoid circular imports (only needed for type checking)
if TYPE_CHECKING:
    from agents.agent_mcts.tree import TreeNode

# board dimensions
BOARD_ROWS = 6
BOARD_COLS = 7
BOARD_SHAPE = (BOARD_ROWS, BOARD_COLS)

BoardPiece = np.int8  # data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0) 
PLAYER1 = BoardPiece(1) 
PLAYER2 = BoardPiece(2) 

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # column (=action) to be played

# NOTE: maybe consider a class for this later to extend functionality?
# for now: SavedState is just a TreeNode (for agent) or None (for human player)
SavedState = Optional["TreeNode"]

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

class MoveStatus(Enum):
    IS_VALID = 1
    WRONG_TYPE = 'Input does not have the correct type (PlayerAction).'
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'


# generator move function type
GenMove = Callable[
    [np.ndarray, BoardPiece, SavedState],  # Arguments for the generate_move function
    tuple[PlayerAction, SavedState]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Initialize and return a new empty game board.

    The board is an ndarray of shape (BOARD_ROWS, BOARD_COLS) and dtype BoardPiece,
    where all positions are set to 0 (NO_PLAYER).

    Returns
    -------
    board : np.ndarray
        An empty game board.
    """
    return np.zeros(BOARD_SHAPE, dtype=BoardPiece)
    
def create_random_game_state(full_board: bool = False) -> np.ndarray:
    """
    Create and return a randomly filled game board.

    The board is an ndarray of shape (BOARD_ROWS, BOARD_COLS) and dtype BoardPiece,
    where each position is randomly assigned to NO_PLAYER (0), PLAYER1 (1), or PLAYER2 (2).

    Parameters
    ----------
    full_board : bool, optional
        Can be used to create a full random board. Default is False.

    Returns
    -------
    board : np.ndarray
        A randomly initialized game board, useful for testing.
    """
    if full_board:
        return np.random.randint(1, 3, size=BOARD_SHAPE, dtype=BoardPiece)
    else:
        return np.random.randint(0, 3, size=BOARD_SHAPE, dtype=BoardPiece)

def pretty_print_board(board: np.ndarray) -> str:
    """
    Convert the game board to a human-readable string representation.

    The piece in board[0, 0] of the array should appear in the lower-left of the
    printed string representation. Here's an example output:

    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |

    Parameters
    ----------
    board : np.ndarray
        The game board as a 2D array.

    Returns
    -------
    str
        The formatted string representation of the board.
    """

    pretty_board = []
    pretty_board.append("|==============|") # top border row

    # 0th row of board should be displayed at the bottom
    for r_i, row in enumerate(np.flipud(board)):
        row_str = create_pretty_row_str(row)
        pretty_board.append(row_str)
    
    pretty_board.append("|==============|") # bottom border row

    # add column indices at bottom
    index_row = "|" + " ".join(str(c) for c in range(BOARD_COLS)) + " |"
    pretty_board.append(index_row)

    return "\n".join(pretty_board)


def create_pretty_row_str(row: np.ndarray) -> str:
    """Create a string representation of a single row of the game board."""
    row_str = "|"

    # each row will contain only integers: 0: no player, 1: player 1, 2: player 2
    for element in row:
        if element == NO_PLAYER:
            row_str += NO_PLAYER_PRINT + " "
        if element == PLAYER1:
            row_str += PLAYER1_PRINT + " "
        if element == PLAYER2:
            row_str += PLAYER2_PRINT + " "
    row_str += "|"
    return row_str


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Convert a string representation of the board (as produced by pretty_print_board)
    back into a NumPy ndarray representing the game board state.

    This is useful for debugging when you have the board as a string and want to
    reconstruct the ndarray.

    Parameters
    ----------
    pp_board : str
        A formatted string representation of the board.

    Returns
    -------
    np.ndarray
        The game board as a 2D array.
    """
    # split the string into rows, ignoring borders and indices
    board_rows = pp_board.split("|\n|")[1:-2]

    board = np.empty(BOARD_SHAPE, dtype=BoardPiece)

    char_to_int = {
        " ": NO_PLAYER,
        "X": PLAYER1,
        "O": PLAYER2
    }
    
    for row_idx, row in enumerate(board_rows):
        # take every second character (skip spaces)
        row_chars = row[::2]
        row_int = [char_to_int[char] for char in row_chars]
        board[row_idx] = row_int

    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> None:
    """
    Apply a player's action to the board by placing their piece in the lowest empty row of the specified column.
    
    Parameters
    ----------
    board : np.ndarray
        The game board array to be modified in place.
    action : PlayerAction
        The column index where the player wants to place their piece.
    player : BoardPiece
        The player’s piece identifier.
    
    Notes
    -----
    The board is modified in place.
    """
    lowest_empty_row = get_lowest_empty_row(board, action)
    board[lowest_empty_row, action] = player


def get_lowest_empty_row(board: np.ndarray, col: PlayerAction) -> int:
    """
    Returns the lowest empty row in the specified column of the board. 
    Return -1 if the column is full.
    """
    column = board[:, col]
    empty_rows = np.where(column == NO_PLAYER)[0]
    if empty_rows.size > 0:
        return int(empty_rows[0])
    else:
        return -1


def connected_four(board: np.ndarray, last_action: PlayerAction, player: BoardPiece) -> bool:
    """
    Check whether the last action by the given player resulted in four 
    connected pieces in a row, column, or diagonal.

    Parameters
    ----------
    board : np.ndarray
        The current state of the game board.
    last_action : PlayerAction
        The column index of the last move.
    player : BoardPiece
        The player who made the last move.

    Returns
    -------
    bool
        True if the last move resulted in four connected pieces, False otherwise.
    """
    row_idx = get_lowest_empty_row(board, last_action) - 1  # row index of the PREVIOUSLY (-> -1) placed piece 
    col_idx = last_action
    
    # check invalid row_idx
    if row_idx >= BOARD_ROWS:
        raise ValueError(f"Invalid row index computed: {row_idx}. Check last_action and board state.")

    diagonal, anti_diagonal = extract_diagonals(board, row_idx, col_idx)

    if four_connected_pieces(board[row_idx,:], player): 
        return True # check horizontal (row)
    if four_connected_pieces(board[:,col_idx], player): 
        return True # check vertical (col)
    if len(diagonal) >= 4 and four_connected_pieces(diagonal, player): 
            return True # check diagonal
    if len(anti_diagonal) >= 4 and four_connected_pieces(anti_diagonal, player): 
            return True # check anti_diagonal 

    return False

def four_connected_pieces(array: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if a given array contains four adjacent pieces of the given player.
    Otherwise, returns False. 
    """
    for idx in range(len(array)-3):
        four_elements_row = array[idx:idx+4]
        # check if all 4 array elements belong to same player
        if np.all(four_elements_row==player):
            return True
    return False


def extract_diagonals(
    board: np.ndarray, 
    row_idx: int, 
    col_idx: PlayerAction
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the diagonal and anti-diagonal passing through a given element 
    (row_idx, col_idx) in the board.

    Parameters
    ----------
    board : np.ndarray
        The current state of the game board.
    row_idx : int
        Row index of the cell.
    col_idx : PlayerAction
        Column index of the cell.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - diagonal: The main diagonal containing the cell.
        - anti_diagonal: The anti-diagonal containing the cell.
    """
    # diagonal (top-left to bottom-right) of given idxs  
    diagonal = np.diag(board, int(col_idx-row_idx))

    # anti-diagonal (top-right to bottom-left) of given idxs
    flipped_board = np.fliplr(board)
    flipped_col_idx = board.shape[1]-1-col_idx # adjust col idx to match flipped board
    anti_diagonal = np.diag(flipped_board, int(flipped_col_idx-row_idx))
    return diagonal, anti_diagonal


def is_full(board: np.ndarray) -> bool:
    """
    Returns True if the given board is fully occupied (no empty spaces),
    otherwise returns False.
    """
    if np.all(board!=NO_PLAYER): return True
    return False


def check_end_state(board: np.ndarray, player: BoardPiece, last_action: PlayerAction) -> GameState:
    """
    Determines the current game state after the given player's last action.

    Parameters
    ----------
    board : np.ndarray
        The current state of the game board.
    player : BoardPiece
        The player who made the last action.
    last_action : PlayerAction
        The last action of the player that led to the current state of the board.

    Returns
    -------
    GameState
        - GameState.IS_WIN if the player has connected four,
        - GameState.IS_DRAW if the board is full,
        - GameState.STILL_PLAYING otherwise.
    """
    if connected_four(board, player, last_action): return GameState.IS_WIN
    elif is_full(board): return GameState.IS_DRAW
    return GameState.STILL_PLAYING


def check_move_status(board: np.ndarray, action: PlayerAction) -> MoveStatus:
    """
    Determines whether a move is valid and, if not, why.

    Parameters
    ----------
    board : np.ndarray
        The current game board.
    action : PlayerAction
        The column index of the move.

    Returns
    -------
    MoveStatus
        - MoveStatus.IS_VALID if the move is valid.
        - MoveStatus.WRONG_TYPE if the column type is incorrect.
        - MoveStatus.OUT_OF_BOUNDS if the column index is out of range.
        - MoveStatus.FULL_COLUMN if the column is already full.
    """
    if not isinstance(action, PlayerAction): return MoveStatus.WRONG_TYPE
    if action >= BOARD_COLS: return MoveStatus.OUT_OF_BOUNDS
    if action < 0: return MoveStatus.OUT_OF_BOUNDS
    if get_lowest_empty_row(board, action) == -1: return MoveStatus.FULL_COLUMN
    return MoveStatus.IS_VALID



def update_saved_state (saved_state: Optional["TreeNode"], action: PlayerAction):
    """
    Updates the saved_state (TreeNode) based on the given action, or returns None if no state is tracked.
    
    Parameters
    ----------
    saved_state : Optional[TreeNode]
        The current node of the search tree, if the player is an agent using a search tree.
    action : PlayerAction
        The action that was just played.
    
    Returns
    -------
    Optional[TreeNode]
        The updated TreeNode corresponding to the played action,
        or None if no search tree is used (e.g., human player).
    """
    if not saved_state: return None # saved state only for TreeNodes
    for child in saved_state.children:
        if child.previous_action == action:
            return child
