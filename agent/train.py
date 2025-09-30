import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import togyzkumalak as tk
from .encode import encode_state
from .tnet import TNET
from .dataset import BoardDataset
from .alphaloss import AlphaLoss

cuda = torch.cuda.is_available()

def train(model: TNET,
          dataset: BoardDataset,
          *,
          epochs=50,
          lr=1e-4,
          lr_step=5000,
          gamma=0.2,
          batch_size=64,
          weight_decay=1e-4,
          grad_clip=1.0,
          log_every=10,
          early_window=16,
          early_tol=0.01,
          checkpoint_path=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=torch.cuda.is_available(),
                        collate_fn=BoardDataset.collate, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=gamma)

    losses_epoch = []
    print(f"[{time.strftime('%H:%M:%S')}] [train] start")

    for epoch in range(epochs):
        t0 = time.time()
        sum_loss = sum_pl = sum_vl = 0.0
        batches = 0

        for i, (x, pi_t, v_t) in enumerate(loader, 1):
            x, pi_t, v_t = x.to(device), pi_t.to(device), v_t.to(device)

            opt.zero_grad(set_to_none=True)
            logits, v_pred = model(x)

            logp = F.log_softmax(logits, dim=-1)
            policy_loss = F.kl_div(logp, pi_t.clamp_min(1e-8), reduction='batchmean')
            value_loss  = F.mse_loss(v_pred.squeeze(-1), v_t)
            loss = policy_loss + value_loss

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            sum_loss += loss.item()
            sum_pl   += policy_loss.item()
            sum_vl   += value_loss.item()
            batches  += 1

            if i % log_every == 0:
                pred_idx = logits[0].argmax().item()
                targ_idx = pi_t[0].argmax().item()
                now = time.strftime('%H:%M:%S')
                print(f"[{now}] [ep {epoch+1}] step {i}: "
                      f"loss={sum_loss/log_every:.4f}  "
                      f"pl={sum_pl/log_every:.4f}  vl={sum_vl/log_every:.4f}  "
                      f"π(t,p)=({targ_idx},{pred_idx})  "
                      f"v(t,p)=({v_t[0].item():+.3f},{v_pred[0].item():+.3f})")
                sum_loss = sum_pl = sum_vl = 0.0

        scheduler.step()
        epoch_loss = (sum_loss if batches % log_every else 0.0) / max(1, (batches % log_every or log_every))
        avg_epoch  = (epoch_loss if batches < log_every else 0.0)  # хвост последнего блока
        losses_epoch.append((avg_epoch,))  # при желании храни среднее по эпохе

        dt = time.time() - t0
        now = time.strftime('%H:%M:%S')
        print(f"[{now}] [ep {epoch+1}] done in {dt:.1f}s, lr={scheduler.get_last_lr()[0]:.2e}")

        if len(losses_epoch) > early_window:
            # простая проверка стагнации по последним/прошлым окнам (можно заменить на метрику)
            recent = sum(x[0] for x in losses_epoch[-4:-1]) / 3.0
            past   = sum(x[0] for x in losses_epoch[-early_window-1:-early_window-4:-1]) / 3.0
            if math.isfinite(recent) and math.isfinite(past) and abs(recent - past) <= early_tol:
                print(f"[{time.strftime('%H:%M:%S')}] [train] early stop at epoch {epoch+1}")
                break

        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)

    print(f"[{time.strftime('%H:%M:%S')}] [train] done")