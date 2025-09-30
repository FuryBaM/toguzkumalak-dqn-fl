import torch
import torch.nn.functional as F

class AlphaLoss(torch.nn.Module):
    def __init__(self, w_policy=1.0, w_value=1.0):
        super().__init__()
        self.wp = w_policy
        self.wv = w_value

    def forward(self, value_pred, logits, value_t, pi_t):
        # value_t: [B] в [-1,1]; pi_t: [B,9] распределение из MCTS
        logp = F.log_softmax(logits, dim=-1)               # предсказание
        pi_t = pi_t.clamp_min(1e-8)                        # числ. устойчивость

        policy_loss = F.kl_div(logp, pi_t, reduction='batchmean')  # H(pi_t, p)
        value_loss  = F.mse_loss(value_pred.squeeze(-1), value_t)

        return self.wp * policy_loss + self.wv * value_loss