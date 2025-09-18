from dataclasses import dataclass, field
from typing import List

INF = 10**6
WHITE, BLACK = 0, 1
GAME_CONTINUE, GAME_WHITE_WIN, GAME_BLACK_WIN, GAME_DRAW = -1, 0, 1, 2
GOAL = 82

@dataclass
class GameState:
    board: List[int] = field(default_factory=lambda: [9]*18)
    white_score: int = 0
    black_score: int = 0
    tuzdyk1: int = -1   # индекс на стороне чёрных (9..17) или -1
    tuzdyk2: int = -1   # индекс на стороне белых (0..8) или -1
    player: int = WHITE