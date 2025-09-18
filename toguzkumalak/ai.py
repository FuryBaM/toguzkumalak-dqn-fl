import math, time
from .state import *
from .rules import make_move, checkWinner
from .eval import evaluate

def gen_moves(state: GameState):
    start = 0 if state.player == WHITE else 9
    return [i for i in range(9) if state.board[start + i] > 0]

def clone_state(state: GameState) -> GameState:
    return GameState(list(state.board), state.white_score, state.black_score,
                     state.tuzdyk1, state.tuzdyk2, state.player)

def tt_key(state: GameState):
    return (tuple(state.board), state.white_score, state.black_score,
            state.tuzdyk1, state.tuzdyk2, state.player)

def order_moves(state: GameState, moves, tt_best=None):
    scored = []
    for m in moves:
        s2 = clone_state(state); make_move(s2, m)
        sc = evaluate(s2); boost = 10_000 if tt_best is not None and m == tt_best else 0
        scored.append((-(sc + boost), m))
    scored.sort()
    return [m for _, m in scored]

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
class TTEntry:
    __slots__ = ("depth","value","flag","best")
    def __init__(self, depth, value, flag, best): self.depth=depth; self.value=value; self.flag=flag; self.best=best

def alphabeta(state: GameState, depth: int, alpha: int, beta: int, tt: dict, start_time: float, time_limit_s: float):
    if time_limit_s is not None and (time.perf_counter() - start_time) >= time_limit_s: raise TimeoutError
    if checkWinner(state) != GAME_CONTINUE or depth == 0: return evaluate(state), None

    key = tt_key(state); tt_best = None
    if key in tt:
        ent = tt[key]
        if ent.depth >= depth:
            if ent.flag == TT_EXACT: return ent.value, ent.best
            if ent.flag == TT_LOWER and ent.value > alpha: alpha = ent.value
            elif ent.flag == TT_UPPER and ent.value < beta: beta = ent.value
            if alpha >= beta: return ent.value, ent.best
        tt_best = ent.best

    best_move = None
    moves = order_moves(state, gen_moves(state), tt_best)
    if not moves: return evaluate(state), None

    maximizing = (state.player == WHITE)
    best_val = -math.inf if maximizing else math.inf

    if maximizing:
        for m in moves:
            s2 = clone_state(state); make_move(s2, m)
            val, _ = alphabeta(s2, depth-1, alpha, beta, tt, start_time, time_limit_s)
            if val > best_val: best_val, best_move = val, m
            if best_val > alpha: alpha = best_val
            if alpha >= beta: break
    else:
        for m in moves:
            s2 = clone_state(state); make_move(s2, m)
            val, _ = alphabeta(s2, depth-1, alpha, beta, tt, start_time, time_limit_s)
            if val < best_val: best_val, best_move = val, m
            if best_val < beta: beta = best_val
            if alpha >= beta: break

    old = tt.get(key)
    if old is None or old.depth <= depth:
        tt[key] = TTEntry(depth, best_val, TT_EXACT, best_move)
    return best_val, best_move

def find_best_move(state: GameState, max_depth: int = 8, time_limit_s: float = 1.5):
    tt = {}; best_move = None; best_val = None; start = time.perf_counter()
    for d in range(1, max_depth+1):
        try:
            val, mv = alphabeta(state, d, -math.inf, math.inf, tt, start, time_limit_s)
            if mv is not None: best_move, best_val = mv, val
        except TimeoutError:
            break
    return best_move, best_val

def ai_turn(state: GameState, max_depth=8, time_limit_s=1.5):
    mv, val = find_best_move(state, max_depth=max_depth, time_limit_s=time_limit_s)
    if mv is None: print("Нет допустимых ходов."); return False
    side = "Белые" if state.player == WHITE else "Чёрные"
    print(f"{side} ИИ выбирает лунку: {mv+1}  | оценка: {val}")
    make_move(state, mv); return True