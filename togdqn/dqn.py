# DQN для тогыз-кумалак (dense-вектор)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# размер входа: 18 лунок + 2 очка + 1 чей ход
INPUT_SIZE = 21
N_ACTIONS = 9  # локальные лунки 0..8

class InputLayers(nn.Module):
    def __init__(self, in_dim=INPUT_SIZE, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
    def forward(self, x):
        x = F.elu(self.fc1(x))
        return x

class BoardFeatures(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
    def forward(self, x):
        residual = x
        out = F.elu(self.fc1(x))
        out = F.elu(self.fc2(out))
        out += residual
        return out

class OutputLayers(nn.Module):
    def __init__(self, hidden=256, dense_size=512):
        super().__init__()
        self.ln1 = nn.Linear(hidden, dense_size)
        self.advantage = nn.Linear(dense_size, N_ACTIONS)
        self.value = nn.Linear(dense_size, 1)
    def forward(self, features):
        x = F.elu(self.ln1(features))
        adv = self.advantage(x)
        val = self.value(x)
        # dueling формула
        return val + adv - adv.mean(dim=1, keepdim=True)

class DQN(nn.Module):
    def __init__(self, in_dim=INPUT_SIZE, hidden=256, num_blocks=4):
        super().__init__()
        self.loss = 0.0
        self.inp = InputLayers(in_dim, hidden)
        self.blocks = nn.ModuleList(BoardFeatures(hidden) for _ in range(num_blocks))
        self.outblock = OutputLayers(hidden)

    def forward(self, s):
        x = self.inp(s)
        for blk in self.blocks:
            x = blk(x)
        return self.outblock(x)

    def select_action(self, state_vec, mask, eps=0.0, device="cpu", sigma=0.05):
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        if np.random.rand() < eps and mask_t.any():
            legal = torch.nonzero(mask_t).flatten().cpu().numpy()
            return int(np.random.choice(legal))
        with torch.no_grad():
            if state_vec.dim() == 1: state_vec = state_vec.unsqueeze(0)
            q = self.forward(state_vec.to(device))[0]           # (9,)
            if sigma > 0:
                q = q + torch.randn_like(q) * sigma             # N(0, sigma)
            q[~mask_t] = -1e9
            return int(torch.argmax(q).item())

        
def train_dueling_dqn(model, target_net, memory, optimizer, batch_size, gamma, alpha_future=0.2):
    if len(memory) < batch_size:
        return

    device = next(model.parameters()).device
    s,a,ns,r,d = zip(*memory.sample(batch_size))

    state_batch      = torch.stack(s).to(device)
    next_state_batch = torch.stack(ns).to(device)
    action_batch     = torch.tensor(a, dtype=torch.long,    device=device)
    reward_batch     = torch.tensor(r, dtype=torch.float32, device=device)
    done_batch       = torch.tensor(d, dtype=torch.float32, device=device)

    # Q(s,·), Q(s′,·)
    q_values            = model(state_batch)
    next_q_values       = model(next_state_batch)       # online
    next_q_state_values = target_net(next_state_batch)  # target

    q_sa      = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
    next_idx  = next_q_values.argmax(1, keepdim=True)
    next_q    = next_q_state_values.gather(1, next_idx).squeeze(1)
    expected_q_value = reward_batch + gamma * next_q * (1.0 - done_batch)

    # common loss (по всей матрице, как в оригинале)
    target_q_values = q_values.detach().clone()
    target_q_values.scatter_(1, action_batch.unsqueeze(1), expected_q_value.unsqueeze(1))
    common_loss = F.smooth_l1_loss(q_values, target_q_values)
    mask = (1.0 - done_batch).view(-1,1)
    # future loss
    future_loss = F.smooth_l1_loss(next_q_values*mask, next_q_state_values*mask)

    loss = (1.0 - alpha_future) * common_loss + alpha_future * future_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    model.loss = float(loss.detach())