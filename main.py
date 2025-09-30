# train_az.py
import os, time, torch, togyzkumalak as tk
from agent.tnet import TNET
from agent.dataset import BoardDataset
from agent.selfplay import selfplay
from agent.train import train

DATASET_PATH = "experiences.pkl"
CHECKPOINT_DIR = "ckpts"
BEST_PATH = os.path.join(CHECKPOINT_DIR, "best.pth")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

GAMES_PER_ITER   = 50
TRAIN_EPOCHS     = 4
BATCH_SIZE       = 256
LR               = 1e-3
ARENA_GAMES      = 200
PROMOTE_THRESH   = 0.55  # winrate кандидата против best

def append_and_trim_dataset(path: str, ds_new: BoardDataset, max_size: int) -> BoardDataset:
    if os.path.exists(path):
        ds_old = BoardDataset.from_pickle(path)
        xs = ds_old.x + ds_new.x
        pis = ds_old.pi + ds_new.pi
        vs = ds_old.v + ds_new.v
    else:
        xs, pis, vs = ds_new.x, ds_new.pi, ds_new.v
    if len(xs) > max_size:
        xs, pis, vs = xs[-max_size:], pis[-max_size:], vs[-max_size:]
    ds = BoardDataset(xs, pis, vs)
    ds.to_pickle(path)
    return ds

def make_search_params_eval() -> tk.SearchParams:
    p = tk.SearchParams()
    p.num_reads = 800
    p.cpuct = 1.5
    p.self_play = False        # без Dirichlet в арене
    p.dirichlet_eps = 0.0
    p.dirichlet_alpha = 0.0
    p.temperature = 0.0        # жёсткий argmax по посещениям
    return p

def make_search_params_selfplay() -> tk.SearchParams:
    p = tk.SearchParams()
    p.num_reads = 800
    p.cpuct = 1.5
    p.self_play = True
    p.dirichlet_eps = 0.25
    p.dirichlet_alpha = 0.3
    p.temperature = 1.0
    return p

@torch.no_grad()
def predictor_from(model: TNET, device):
    from agent.encode import encode_state
    def predictor(state: tk.GameState):
        x = encode_state(state).unsqueeze(0).to(device)
        logits, v = model(x)
        priors = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return priors, float(v.item())
    return predictor

@torch.no_grad()
def arena_eval(model_a: TNET, model_b: TNET, params: tk.SearchParams, games: int, device) -> tuple[int,int,int]:
    pa = predictor_from(model_a.eval(), device)
    pb = predictor_from(model_b.eval(), device)
    wa = wb = dr = 0
    for g in range(games):
        s = tk.GameState()
        turn_a = (g % 2 == 0)  # чередуем первый ход
        while True:
            res = tk.check_winner(s)
            if res != tk.GameResult.GAME_CONTINUE:
                if res == tk.GameResult.GAME_WHITE_WIN:
                    # победил тот, кто делал последний ход белыми
                    # определим по очередности хода
                    # проще: пересчёт по цветам не нужен — счёт пойдёт по модели, делавшей последний ход
                    pass
                # итог строго по текущему игроку не нужен; проще вести по активной модели
                break
            pred = pa if turn_a else pb
            a, _ = tk.mcts_search(s, pred, params)
            if a < 0:  # падение в терминал
                break
            tk.make_move(s, a)
            turn_a = not turn_a
            # цикл дальше

        res = tk.check_winner(s)
        if res == tk.GameResult.GAME_WHITE_WIN:
            # кто сделал последний ход белыми?
            # Если количество сделанных ходов нечётное, первым ходил A при g%2==0
            # Проще вести через локальный флаг: последний ход сделал not turn_a
            winner_is_a = (not turn_a) == (g % 2 == 0)
        elif res == tk.GameResult.GAME_BLACK_WIN:
            winner_is_a = (not turn_a) != (g % 2 == 0)
        else:
            winner_is_a = None

        if winner_is_a is None:
            dr += 1
        elif winner_is_a:
            wa += 1
        else:
            wb += 1
    return wa, wb, dr

def load_or_clone_best(model: TNET, device):
    best = TNET().to(device)
    if os.path.exists(BEST_PATH):
        best.load_state_dict(torch.load(BEST_PATH, map_location=device))
        best.eval()
        print("[arena] loaded best.pth")
    else:
        best.load_state_dict(model.state_dict())
        best.eval()
        torch.save(best.state_dict(), BEST_PATH)
        print("[arena] initialized best.pth from fresh model")
    return best

def main():
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TNET().to(device).train()
    best_model = load_or_clone_best(model, device)

    sp_params = make_search_params_selfplay()
    ev_params = make_search_params_eval()

    for it in range(1, 10_000):
        t0 = time.time()

        # 1) Самоигра
        selfplay(model, DATASET_PATH, sp_params, num_games=GAMES_PER_ITER)

        # 2) Датасет окно
        ds_new = BoardDataset.from_pickle(DATASET_PATH)
        ds = append_and_trim_dataset(DATASET_PATH, ds_new, MAX_DATASET_SIZE)

        # 3) Обучение
        train(model, ds, epochs=TRAIN_EPOCHS, lr=LR, batch_size=BATCH_SIZE,
              weight_decay=1e-4, grad_clip=1.0, log_every=10)

        # 4) Чекпойнт кандидата
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"tnet_iter{it:05d}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # 5) Арена: candidate vs best
        model.eval()
        wa, wb, dr = arena_eval(model, best_model, ev_params, ARENA_GAMES, device)
        total = max(1, wa + wb + dr)
        winrate = wa / max(1, (wa + wb))
        print(f"[arena] A_vs_B: A(wins)={wa} B(wins)={wb} draws={dr} -> winrate={winrate:.3f}")

        # 6) Продвижение
        if winrate > PROMOTE_THRESH:
            torch.save(model.state_dict(), BEST_PATH)
            best_model.load_state_dict(model.state_dict())
            best_model.eval()
            print(f"[arena] promoted: winrate {winrate:.3f} > {PROMOTE_THRESH:.2f}")
        else:
            print(f"[arena] rejected: winrate {winrate:.3f} <= {PROMOTE_THRESH:.2f}")

        dt = time.time() - t0
        print(f"[iter {it}] done in {dt:.1f}s, dataset={len(ds)} samples, ckpt={ckpt_path}")

if __name__ == "__main__":
    main()
