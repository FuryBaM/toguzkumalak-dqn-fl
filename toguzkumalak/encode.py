import torch, numpy as np
from .state import *

INPUT_SIZE = 23  # 18 pits + 2 scores + 2 tuzdyk flags + 1 player
N_ACTIONS = 9    # локальные ходы 0..8

def encode_state(state) -> torch.Tensor:
    # board: копия с заменой туздыков на -1
    pits = torch.tensor(state.board, dtype=torch.float32)
    if state.tuzdyk1 != -1:
        pits[state.tuzdyk1] = -1.0
    if state.tuzdyk2 != -1:
        pits[state.tuzdyk2] = -1.0

    # нормализация (камни делим на 162, а -1 оставляем как есть)
    pits = torch.where(pits >= 0, pits / 162.0, pits)

    # очки нормируем
    scores = torch.tensor([state.white_score, state.black_score], dtype=torch.float32) / 162.0
    # чей ход
    pl = torch.tensor([float(state.player)], dtype=torch.float32)

    return torch.cat([pits, scores, pl])  # 18 + 2 + 1 = 21 признаков

def legal_mask(state):
    if state.player == WHITE:
        row = state.board[0:9]
    else:
        row = state.board[9:18]
    return np.array([1 if v > 0 else 0 for v in row], dtype=np.int8)

def outcome_sign_for_step(player_at_step, final_result):  
    # final_result: +1 белые, -1 чёрные, 0 ничья
    if final_result == 0:
        return 0
    return final_result if player_at_step == WHITE else -final_result