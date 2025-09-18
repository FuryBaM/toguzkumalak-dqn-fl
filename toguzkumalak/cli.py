from .state import *
from .rules import render_board, make_move, checkWinner
from .eval import evaluate
from .ai import ai_turn

def run_game():
    state = GameState(); render_board(state)
    while True:
        print(evaluate(state))
        res = checkWinner(state)
        if res != GAME_CONTINUE:
            print("Результат:", "победа белых" if res==GAME_WHITE_WIN else "победа чёрных" if res==GAME_BLACK_WIN else "ничья")
            break
        name = "Белые" if state.player == WHITE else "Чёрные"
        s = input(f"{name}, выберите лунку [1-9] или q: ").strip().lower()
        if s == "q": print("Выход."); break
        if not s.isdigit(): print("Ошибка."); continue
        pit = int(s)-1
        if not (0 <= pit <= 8) or not make_move(state, pit):
            print("Ход недопустим."); continue
        render_board(state)

def run_game_vs_ai(human_plays_white=True, max_depth=8, time_limit_s=1.5):
    state = GameState(); render_board(state)
    while True:
        print(evaluate(state))
        res = checkWinner(state)
        if res != GAME_CONTINUE:
            print("Результат:", "победа белых" if res==GAME_WHITE_WIN else "победа чёрных" if res==GAME_BLACK_WIN else "ничья")
            break
        human_turn = (state.player==WHITE and human_plays_white) or (state.player==BLACK and not human_plays_white)
        if human_turn:
            name = "Белые" if state.player == WHITE else "Чёрные"
            s = input(f"{name}, выберите лунку [1-9] или q: ").strip().lower()
            if s == "q": print("Выход."); break
            if not s.isdigit(): print("Ошибка."); continue
            pit = int(s)-1
            if not (0 <= pit <= 8) or not make_move(state, pit):
                print("Ход недопустим."); continue
            render_board(state)
        else:
            if not ai_turn(state, max_depth=max_depth, time_limit_s=time_limit_s):
                print("Нет допустимых ходов."); break
            render_board(state)