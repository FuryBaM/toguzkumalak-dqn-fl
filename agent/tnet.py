import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import togyzkumalak as tk
from agent.encode import encode_state

ACTION_SIZE = 9
cuda = torch.cuda.is_available()

# ширина и число блоков
WIDTH = 128
BLOCKS = 8

class InputLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(21, WIDTH)
        self.ln = nn.LayerNorm(WIDTH)
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        return F.silu(self.ln(self.fc(x)))

class BoardFeatures(nn.Module):
    def __init__(self, w=WIDTH):
        super().__init__()
        self.fc1 = nn.Linear(w, w); self.ln1 = nn.LayerNorm(w)
        self.fc2 = nn.Linear(w, w); self.ln2 = nn.LayerNorm(w)
    def forward(self, x):
        y = F.silu(self.ln1(self.fc1(x)))
        y = self.ln2(self.fc2(y))
        return F.silu(x + y)

class OutputLayers(nn.Module):
    def __init__(self, w=WIDTH):
        super().__init__()
        self.p1 = nn.Linear(w, 64); self.p2 = nn.Linear(64, ACTION_SIZE)
        self.v1 = nn.Linear(w, 64); self.v2 = nn.Linear(64, 1)
    def forward(self, x):
        logits = self.p2(F.silu(self.p1(x)))
        value  = torch.tanh(self.v2(F.silu(self.v1(x))))
        return logits, value

class TNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputblock = InputLayers()
        self.res = nn.ModuleList([BoardFeatures() for _ in range(BLOCKS)])
        self.outblock = OutputLayers()
    def forward(self, s):
        s = self.inputblock(s)
        for blk in self.res: s = blk(s)
        return self.outblock(s)
    
    @torch.no_grad()
    def act(self, state: tk.GameState):
        self.eval()
        x = encode_state(state).unsqueeze(0)
        x = x.cuda() if cuda else x
        self.to(x.device)
        logits, _ = self(x)
        logits = logits.squeeze(0)
        legal_idx = tk.gen_moves(state)  # список индексов 0..8
        if not legal_idx:
            return -1
        mask = torch.full_like(logits, float("-inf"))
        mask[legal_idx] = 0.0
        return int(torch.argmax(logits + mask).item())
    
    def save(self, path):
        with torch.no_grad():
            # Сохраняем только веса модели для дальнейшего обучения
            torch.save(self.state_dict(), path)
            
    def load(self, path):
        device = torch.device('cuda' if cuda else 'cpu')
        ckpt = torch.load(path, map_location=device)
        self.load_state_dict(ckpt)
        self.to(device).eval()