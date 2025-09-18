from .state import *

def isPitEmpty(state: GameState, x: int) -> bool:
    if not (0 <= x < 18): raise IndexError()
    return state.board[x] == 0

def isValidMove(state: GameState, x: int) -> bool:
    return (((state.player == WHITE and 0 <= x < 9) or
             (state.player == BLACK and 9 <= x < 18))
            and not isPitEmpty(state, x))

def isRowEmpty(state: GameState, player: int) -> bool:
    s = 9 * player; e = s + 9
    return all(v == 0 for v in state.board[s:e])

def checkWinner(state: GameState) -> int:
    if state.white_score == GOAL-1 and state.black_score == GOAL-1: return GAME_DRAW
    if (state.white_score >= GOAL) or (state.black_score < GOAL and isRowEmpty(state, BLACK)): return GAME_WHITE_WIN
    if (state.black_score >= GOAL) or (state.white_score < GOAL and isRowEmpty(state, WHITE)): return GAME_BLACK_WIN
    return GAME_CONTINUE

def make_tuzdyk(state: GameState, x: int):
    if state.board[x] == 3 and (x % 9) != 8:
        if state.player == WHITE and 9 <= x < 18 and state.tuzdyk1 == -1:
            if state.tuzdyk2 == -1 or (state.tuzdyk2 % 9) != (x % 9):
                state.tuzdyk1 = x; state.white_score += 3; state.board[x] = 0
        elif state.player == BLACK and 0 <= x < 9 and state.tuzdyk2 == -1:
            if state.tuzdyk1 == -1 or (state.tuzdyk1 % 9) != (x % 9):
                state.tuzdyk2 = x; state.black_score += 3; state.board[x] = 0

def take_pits(state: GameState, x: int):
    if state.board[x] % 2 == 0:
        if state.player == WHITE and 9 <= x < 18:
            state.white_score += state.board[x]; state.board[x] = 0
        elif state.player == BLACK and 0 <= x < 9:
            state.black_score += state.board[x]; state.board[x] = 0

def make_move(state: GameState, pit_local: int) -> bool:
    x = pit_local + 9 * state.player
    if not isValidMove(state, x) or checkWinner(state) != GAME_CONTINUE: return False

    # ВАЖНО: взять все камни
    stones = state.board[x]; state.board[x] = 0

    for _ in range(stones):
        x = (x + 1) % 18
        state.board[x] += 1

    if state.tuzdyk1 != -1:
        state.white_score += state.board[state.tuzdyk1]; state.board[state.tuzdyk1] = 0
    if state.tuzdyk2 != -1:
        state.black_score += state.board[state.tuzdyk2]; state.board[state.tuzdyk2] = 0

    make_tuzdyk(state, x)
    take_pits(state, x)

    state.player ^= 1
    return True

def render_board(state: GameState):
    W = 5
    def cell(i: int) -> str:
        n = state.board[i]; mark = ""
        if i == state.tuzdyk1: mark = "*W"
        if i == state.tuzdyk2: mark = "*B"
        return f"{(str(n)+mark):>{W}}"

    top_idx = list(range(17, 9-1, -1))
    bot_idx = list(range(0, 9))

    top_labels = " ".join(f"{i:>{W}}" for i in range(9, 0, -1))
    bot_labels = " ".join(f"{i:>{W}}" for i in range(1, 10))

    top = " ".join(cell(i) for i in top_idx)
    bot = " ".join(cell(i) for i in bot_idx)

    print("\n   ЧЁРНЫЕ"); print("   " + top_labels); print("   " + top)
    print("   " + bot);  print("   " + bot_labels); print("   БЕЛЫЕ\n")
    print(f"Счёт: White={state.white_score}  Black={state.black_score}")
    print(f"Ход: {'White' if state.player == WHITE else 'Black'}")
    print(f"Туздык(W): {state.tuzdyk1 if state.tuzdyk1!=-1 else '—'}   "
          f"Туздык(B): {state.tuzdyk2 if state.tuzdyk2!=-1 else '—'}")
    print("-"*72)