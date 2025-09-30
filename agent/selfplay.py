import os, torch, togyzkumalak as tk
from .tnet import TNET
from .dataset import BoardDataset
from .encode import encode_state, outcome_sign_for_step

import os, time
import torch
import togyzkumalak as tk
from .encode import encode_state, outcome_sign_for_step
from .dataset import BoardDataset
from .tnet import TNET

@torch.no_grad()
def selfplay(model: TNET, path: str, params: tk.SearchParams, num_games: int):
    device = next(model.parameters()).device
    model.eval()

    # JIT-инферер (freeze + trace один раз)
    ex = torch.randn(1, 21, device=device)
    with torch.inference_mode():
        jit_infer = torch.jit.freeze(torch.jit.trace(model, ex, strict=False)).to(device)
        jit_infer.eval()

    def predictor(state: tk.GameState):
        x = encode_state(state).unsqueeze(0).to(device)
        # смешанная точность на GPU (ускоряет)
        if torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, v = jit_infer(x)
        else:
            logits, v = jit_infer(x)
        priors = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return priors, float(v.item())

    all_states, all_pis, all_vs = [], [], []

    for g in range(1, num_games + 1):
        start_t = time.time()
        print(f"[selfplay] game {g}/{num_games} started at {time.strftime('%H:%M:%S')}")

        s = tk.GameState()
        states, pis, players = [], [], []
        t = 0
        while True:
            res = tk.check_winner(s)
            if res != tk.GameResult.GAME_CONTINUE:
                z = 1.0 if res == tk.GameResult.GAME_WHITE_WIN else -1.0 if res == tk.GameResult.GAME_BLACK_WIN else 0.0
                vs = [outcome_sign_for_step(p, 1 if z > 0 else -1 if z < 0 else 0) for p in players]
                all_states += states
                all_pis    += pis
                all_vs     += vs
                break

            params.temperature = 1.0 if t < 15 else 0.0
            params.self_play = True

            a, pi = tk.mcts_search(s, predictor, params)
            states.append(encode_state(s))
            pis.append(torch.tensor(pi, dtype=torch.float32))
            players.append(int(s.player))

            if a < 0:
                z = 0.0
                vs = [0.0] * len(players)
                all_states += states; all_pis += pis; all_vs += vs
                break

            tk.make_move(s, a)
            t += 1

        dt = time.time() - start_t
        # итог: счёт, победитель, ходы
        res = tk.check_winner(s)
        ws, bs = int(s.white_score), int(s.black_score)
        if res == tk.GameResult.GAME_WHITE_WIN:
            outcome = "WHITE win"
        elif res == tk.GameResult.GAME_BLACK_WIN:
            outcome = "BLACK win"
        else:
            outcome = "DRAW"
        print(f"[selfplay] game {g}/{num_games} finished at {time.strftime('%H:%M:%S')} "
              f"({dt:.1f}s, {t} moves) score W:{ws} B:{bs} -> {outcome}")

    ds = BoardDataset(all_states, all_pis, all_vs)

    MAX_SIZE = 200_000
    if os.path.exists(path):
        old = BoardDataset.from_pickle(path)
        new_x = old.x + ds.x
        new_pi = old.pi + ds.pi
        new_v = old.v + ds.v
        if len(new_x) > MAX_SIZE:
            new_x, new_pi, new_v = new_x[-MAX_SIZE:], new_pi[-MAX_SIZE:], new_v[-MAX_SIZE:]
        ds = BoardDataset(new_x, new_pi, new_v)

    ds.to_pickle(path)
    print(f"[selfplay] dataset saved to {path}, size={len(ds)}")
