# eval_vs_minimax.py
import torch
from .state import GameState, WHITE, BLACK, GAME_CONTINUE, GAME_WHITE_WIN, GAME_BLACK_WIN
from .rules import make_move, checkWinner, render_board
from .ai import find_best_move
from .encode import encode_state, legal_mask  # твои функции кодирования

def play_one(dqn, dqn_as_white=True, max_depth=8, time_limit_s=1.5, device=None, render=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dqn.to(device).eval()

    s = GameState()
    while True:
        if render:
            render_board(s)

        res = checkWinner(s)
        if res != GAME_CONTINUE:
            if res == GAME_WHITE_WIN: return +1
            if res == GAME_BLACK_WIN: return -1
            return 0

        dqn_turn = (s.player == WHITE and dqn_as_white) or (s.player == BLACK and not dqn_as_white)
        if dqn_turn:
            sv = encode_state(s)              # (INPUT_SIZE,)
            mask = legal_mask(s)              # (9,)
            a = dqn.select_action(sv, mask, eps=0.0, device=device)
            ok = make_move(s, a)
            if not ok:  # на всякий
                # выберем первый легальный
                legal = [i for i, m in enumerate(mask) if m]
                if not legal:  # совсем нет ходов
                    # принудительная оценка финала
                    return +1 if dqn_as_white else -1
                make_move(s, legal[0])
        else:
            mv, _ = find_best_move(s, max_depth=max_depth, time_limit_s=time_limit_s)
            if mv is None:
                # у минимакса нет хода: считаем победу стороны соперника
                return +1 if dqn_as_white else -1
            make_move(s, mv)

def play_match(dqn, games=50, dqn_as_white=True, max_depth=8, time_limit_s=1.5, device=None, render_every=10):
    w = d = l = 0
    for g in range(1, games+1):
        res = play_one(dqn, dqn_as_white=(dqn_as_white if g % 2 else not dqn_as_white),
                       max_depth=max_depth, time_limit_s=time_limit_s, device=device,
                       render=(render_every and g % render_every == 0))
        # res: +1 если победили белые
        if (res == +1 and (dqn_as_white if g % 2 else not dqn_as_white)) or \
           (res == -1 and not (dqn_as_white if g % 2 else not dqn_as_white)):
            w += 1
        elif res == 0:
            d += 1
        else:
            l += 1
    print(f"DQN vs Minimax: {w}-{d}-{l}  (W-D-L)  | games={games}, depth={max_depth}")
    return w, d, l
