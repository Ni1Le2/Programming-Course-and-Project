# The game_utils module is going to contain code of general relevance
# for playing the game and for the agents you will implement.

from typing import Callable, Optional, Any
from enum import Enum
import numpy as np

BOARD_ROWS = 6
BOARD_COLS = 7
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

class MoveStatus(Enum):
    IS_VALID = 1
    WRONG_TYPE = 'Input does not have the correct type (PlayerAction).'
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=BoardPiece)
    
def create_random_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    Useful for testing.
    """
    return np.random.randint(0,3, size=BOARD_SHAPE)

def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] of the array should appear in the lower-left in the printed string representation. 
    Here's an example output, note that we use PLAYER1_Print to represent PLAYER1 and 
    PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    pretty_board = ""
    
    # top border row
    pretty_board += "|==============|\n"

    # board[0,0] should appear in lower left, so 0th row should be displayed at the bottom,
    # 0th col should be displayed at the front
    # -> flip board matrix upside down
    for r_i, row in enumerate(np.flipud(board)):
        pretty_board += "|"

        # each row will contain only integers: 0: no player, 1/2: player 1/2
        for element in row:
            if element == NO_PLAYER:
                pretty_board += NO_PLAYER_PRINT + " "
            if element == PLAYER1:
                pretty_board += PLAYER1_PRINT + " "
            if element == PLAYER2:
                pretty_board += PLAYER2_PRINT + " "
        pretty_board += "|\n"
    
    # bottom border row
    pretty_board += "|==============|\n"

    # row with number indices
    pretty_board += "|"
    for c_i in range(BOARD_COLS):
        pretty_board += str(c_i) + " "
    pretty_board += "|"
    return pretty_board

def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    # extract rows from board
    board_rows = pp_board.split("|\n|")[1:-2]
    board = np.empty((BOARD_SHAPE))
    for row_idx, row in enumerate(board_rows):
        # slice every second character as they are spaces for display
        slice_idx = slice(0, len(row), 2)
        row_string = row[slice_idx]
        # use dict to extract numbers from string
        char_to_int = {" ": NO_PLAYER, "X": PLAYER1, "O": PLAYER2}
        row_int = [char_to_int[char] for char in row_string]
        board[row_idx] = row_int
    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Sets board[i, action] = player, where i is the lowest open row. The input 
    board should be modified in place, such that it's not necessary to return 
    something.
    """
    lowest_empty_row = get_lowest_empty_row(board, action)
    board[lowest_empty_row, action] = player
    # moves_made += 1 # would be nice for checking draws, maybe implement later


def get_lowest_empty_row(board, col):
    # remember: board[0,0] is BOTTOM LEFT corner of the board
    column = board[:,col]
    # find first (lowest) row that is not occupied by a player
    if np.any(np.argwhere(column==NO_PLAYER)):
        lowest_empty_row = int(np.argwhere(column==NO_PLAYER)[0])
    # careful: this returns a different data type!
    else: lowest_empty_row = -1 # no empty rows in this column
    return lowest_empty_row


def connected_four(board: np.ndarray, last_action: PlayerAction, player: BoardPiece) -> bool:
    """
    Returns True if the last action of the player resulted in four adjacent pieces 
    equal to `player` arranged in either a horizontal, vertical, or diagonal line. 
    Returns False otherwise.
    """
    # lowest_empty_row -1 as we want the index of the PREVIOUS action
    
    if get_lowest_empty_row(board, last_action):
        row_idx = get_lowest_empty_row(board, last_action)-1
    # take care of full rows -> last piece was placed in last row
    else: row_idx = -1 
    col_idx = last_action 

    # sufficient to check the one row, one column, and two diagonals which the last action affected!
    if horizontal_win_check(board[row_idx,:], player): return True
    elif vertical_win_check(board[:,col_idx], player): return True
    elif diagonal_win_check(board, row_idx, col_idx, player): return True
    else: return False

def horizontal_win_check(board_row, player: BoardPiece):
    for idx in range(len(board_row)-3):
        # look at 4 adjacent elements of the row
        four_elements_row = slice(idx, idx+4, 1)
        # player (int) decides compared array
        if np.all(board_row[four_elements_row]==np.ones((4))*player):
            return True
    return False


def vertical_win_check(board_col, player: BoardPiece):
    for idx in range(len(board_col)-3):
        # look at 4 adjacent elements of the column
        four_elements_col = slice(idx, idx+4, 1)
        # player (int) decides compared array
        if np.all(board_col[four_elements_col]==np.ones((4))*player):
            return True
    return False


def diagonal_win_check(board, row_idx, col_idx, player: BoardPiece):
    # only need to check two diagonals of given idx   
    # took me some time but works now and is clean
    
    diagonal = np.diag(board, int(col_idx-row_idx))
    # flip matrix left/right and change column index accordingly 
    # (i.e. column 0 needs to be adjusted to be the last column of the flipped matrix)
    anti_diagonal = np.diag(np.fliplr(board), int((board.shape[1] - 1 - col_idx) - row_idx))

    if len(diagonal) < 4 :
        return False
    if len(anti_diagonal) < 4:
        return False
    
    # look at all elements from idx to idx+3 for the diagonal
    for idx in range(len(diagonal)-3):
        four_elements_diag = slice(idx, idx+4, 1)
        if np.all(diagonal[four_elements_diag]==np.ones((4))*player):
            return True
    # look at all elements from idx to idx+3 for the anit-diagonal
    for idx in range(len(anti_diagonal)-3):
        four_elements_diag = slice(idx, idx+4, 1)
        if np.all(anti_diagonal[four_elements_diag]==np.ones((4))*player):
            return True   
    return False


def is_draw(board) -> bool:
    """
    Game is played until the very end, i.e. a draw only occurs if all cells are filled.
    """
    # not really necessary to pass player, but may still be helpful?
    if np.all(board!=0): return True
    return False

def check_end_state(board: np.ndarray, player: BoardPiece, last_action: PlayerAction) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player, last_action): return GameState.IS_WIN
    if is_draw(board): return GameState.IS_DRAW
    return GameState.STILL_PLAYING


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:
    """
    Returns a MoveStatus indicating whether a move is accepted as a valid move 
    or not, and if not, why.
    The provided column must be of the correct type (PlayerAction).
    Furthermore, the column must be within the bounds of the board and the
    column must not be full.
    """
    if not isinstance(column, PlayerAction): return MoveStatus.WRONG_TYPE
    if column >= BOARD_COLS: return MoveStatus.OUT_OF_BOUNDS
    if column < 0: return MoveStatus.OUT_OF_BOUNDS
    if get_lowest_empty_row(board, column) == -1: return MoveStatus.FULL_COLUMN
    return MoveStatus.IS_VALID