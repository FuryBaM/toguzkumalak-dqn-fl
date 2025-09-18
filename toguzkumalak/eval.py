from .state import *
from .rules import checkWinner

def legal_moves_count(state: GameState, player: int) -> int:
    s = 9 * player
    return sum(1 for i in range(s, s+9) if state.board[i] > 0)

def evaluate(state: GameState) -> int:
    res = checkWinner(state)
    if res == GAME_WHITE_WIN: return +INF
    if res == GAME_BLACK_WIN: return -INF
    if res == GAME_DRAW: return 0

    pending_white = state.board[state.tuzdyk1] if state.tuzdyk1 != -1 else 0
    pending_black = state.board[state.tuzdyk2] if state.tuzdyk2 != -1 else 0
    material = (state.white_score + pending_white) - (state.black_score + pending_black)

    own_white  = sum(state.board[0:9])
    opp_white  = sum(state.board[9:18])
    SIDE_W = 0.2
    side_mass = SIDE_W * (opp_white - own_white)

    TUZ_BONUS = 9
    tuz = (TUZ_BONUS if state.tuzdyk1 != -1 else 0) - (TUZ_BONUS if state.tuzdyk2 != -1 else 0)

    EVEN_W = 0.3
    evens_black = sum(v for v in state.board[9:18] if v % 2 == 0)
    evens_white = sum(v for v in state.board[0:9]  if v % 2 == 0)
    parity_potential = EVEN_W * (evens_black - evens_white)

    MOB_W = 0.5
    mob = MOB_W * (legal_moves_count(state, WHITE) - legal_moves_count(state, BLACK))

    return int(round(material + tuz + side_mass + parity_potential + mob))