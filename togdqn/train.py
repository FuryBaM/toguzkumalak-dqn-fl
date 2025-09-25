# train.py
import os, torch, torch.optim as optim
import numpy as np
from .dqn import DQN, train_dueling_dqn
from .selfplay import selfplay_episode, ReplayBuffer
from .encode import encode_state, legal_mask
from .state import GameState, WHITE, GAME_CONTINUE, GAME_WHITE_WIN, GAME_BLACK_WIN
from .rules import make_move, checkWinner

def _play_one(a_net, b_net, a_as_white=True, device="cpu", max_moves=600,
              start_random_moves=2, sigma=0.05):
    s = GameState()
    m = 0

    # дебют: k случайных легальных ходов
    for _ in range(start_random_moves):
        if checkWinner(s) != GAME_CONTINUE: break
        mask = legal_mask(s)
        legal = np.flatnonzero(mask)
        if len(legal) == 0: break
        make_move(s, int(np.random.choice(legal)))
        m += 1

    while m < max_moves:
        res = checkWinner(s)
        if res != GAME_CONTINUE:
            if res == GAME_WHITE_WIN: return +1
            if res == GAME_BLACK_WIN: return -1
            return 0
        net = a_net if (s.player == WHITE) == a_as_white else b_net
        sv = encode_state(s); mask = legal_mask(s)
        a = net.select_action(sv, mask, eps=0.0, device=device, sigma=sigma)
        make_move(s, a); m += 1
    return 0

def arena_vs_snapshot(curr, snap, games=40, device="cpu"):
    w = d = l = 0
    for g in range(games):
        a_white = (g % 2 == 0)
        r = _play_one(curr, snap, a_as_white=a_white, device=device,
                      start_random_moves=2, sigma=0.05)
        if (r == +1 and a_white) or (r == -1 and not a_white): w += 1
        elif r == 0: d += 1
        else: l += 1
    winrate = 100.0 * w / max(1, (w+l))
    return w, d, l, winrate

# ── безопасная загрузка весов ─────────────────────────────────────────────────
def load_or_new(model: DQN, path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        return True
    except Exception:
        return False

# ── трейн с поддержкой пути и арены ───────────────────────────────────────────
def train_selfplay(episodes=5000,
                   batch_size=256,
                   gamma=0.995,
                   alpha_future=0.2,
                   eps_start=1.0,
                   eps_end=0.05,
                   eps_decay=3000,
                   target_update=500,
                   device="cuda",
                   model_path:str="",
                   save_path:str="dqn_togyz.pt",
                   snapshot_every:int=200,
                   arena_games:int=20):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DQN().to(device)
    ok = load_or_new(model, model_path)
    if not ok and model_path:
        print("Предупреждение: не удалось загрузить модель. Создаю новую.")

    target_net = DQN().to(device)
    target_net.load_state_dict(model.state_dict())
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    memory = ReplayBuffer(capacity=200_000)

    eps = eps_start
    steps = 0
    snapshot = DQN().to(device)
    snapshot.load_state_dict(model.state_dict())  # стартовый «оппонент»

    for ep in range(1, episodes+1):
        # партия самоигры
        epbuf, result = selfplay_episode(model, eps=eps, device=device)
        epbuf.flush_to(memory, result)

        # несколько апдейтов
        for _ in range(8):
            train_dueling_dqn(model, target_net, memory, optimizer,
                              batch_size, gamma, alpha_future)
            steps += 1
            if steps % target_update == 0:
                target_net.load_state_dict(model.state_dict())

        # eps-decay
        eps = max(eps_end, eps * torch.exp(torch.tensor(-1.0/eps_decay)).item())

        # лог
        if ep % 100 == 0:
            print(f"Эпизод {ep}: буфер={len(memory)}, loss={model.loss:.4f}")

        # арена и обновление снапшота
        if snapshot_every and (ep % snapshot_every == 0):
            # в train_selfplay, вместо простого сравнения:
            w,d,l,wr = arena_vs_snapshot(model, snapshot, games=40, device=device)
            print(f"[АРЕНА ep{ep}] {w}-{d}-{l}  winrate={wr:.1f}%")
            if wr > 60.0:   # значимый порог
                snapshot.load_state_dict(model.state_dict())

    torch.save(model.state_dict(), save_path)
    print(f"Сеть сохранена в {save_path}")
    return model
