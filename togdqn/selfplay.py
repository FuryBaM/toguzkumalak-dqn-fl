import numpy as np
from collections import namedtuple
from .state import GameState, WHITE, GAME_CONTINUE, GAME_WHITE_WIN, GAME_BLACK_WIN, GAME_DRAW
from .rules import make_move, checkWinner
from .encode import encode_state, legal_mask, outcome_sign_for_step  # то что мы обсуждали выше

# переход: (состояние, действие, следующее, награда, done)
Transition = namedtuple("T", "s a ns r d")

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.data = []
        self.pos = 0

    def push(self, t: Transition):
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.data), batch_size, replace=False)
        return [self.data[i] for i in idxs]

    def __len__(self):
        return len(self.data)

class EpisodeBuffer:
    def __init__(self):
        self.steps = []  # (s, a, ns, player, done)

    def add(self, s, a, ns, player, done):
        self.steps.append((s, a, ns, player, done))

    def flush_to(self, replay: ReplayBuffer, final_result: int):
        for (s, a, ns, pl, done) in self.steps:
            r = outcome_sign_for_step(pl, final_result)
            replay.push(Transition(s, a, ns, r, done))
        self.steps.clear()

def final_result_scalar(state: GameState) -> int:
    res = checkWinner(state)
    if res == GAME_WHITE_WIN: return +1
    if res == GAME_BLACK_WIN: return -1
    return 0

def selfplay_episode(policy_net, eps: float, device, max_moves=500):
    """
    Генерация одной партии.
    policy_net – твоя DQN-сеть (выдаёт Q(s)).
    eps – эпсилон для eps-greedy.
    Возвращает EpisodeBuffer и итог (+1/-1/0).
    """

    state = GameState()
    ep = EpisodeBuffer()
    moves = 0

    while moves < max_moves:
        s_enc = encode_state(state)           # (23,)
        mask = legal_mask(state)

        a = policy_net.select_action(s_enc, mask, eps, device)
        pl = state.player                     # чей ход

        ok = make_move(state, a)
        assert ok, "mask/select_action должны исключать нелегальные ходы"
        ns_enc = encode_state(state)
        moves += 1

        res = checkWinner(state)
        done = (res != GAME_CONTINUE) or (moves >= max_moves)
        ep.add(s_enc, a, ns_enc, pl, done)

        if done:
            return ep, final_result_scalar(state)
