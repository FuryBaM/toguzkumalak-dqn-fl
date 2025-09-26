# from togdqn import run_game_vs_ai, run_game
# from togdqn.train import *
# from togdqn.eval_vs_minimax import play_match
from togyzkumalak import *
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_selfplay(model_path="dqn_togyz.pt", save_path="dqn_togyz_v2.pt", snapshot_every=500)
    # model = DQN().to(device)
    # model.load_state_dict(torch.load("dqn_togyz_v2.pt", map_location="cpu"))
    # play_match(model, games=50, dqn_as_white=True, max_depth=6, time_limit_s=2.0, render_every=1)
    game = GameState()
    while check_winner(game) == GAME_CONTINUE:
        render_board(game)
        if game.player == WHITE:
            x = int(input("Enter your move (1-9): "))
            if not is_valid_move(game, x-1):
                print("Invalid move. Try again.")
                continue
            mv = make_move(game, x-1)
            print(f"You played: {mv}")
        else:
            mv = ai_turn(game, max_depth=16, time_limit_s=4.0)
            print(f"AI played: {mv}")
    render_board(game)
    winner = check_winner(game)
    if winner == GAME_DRAW:
        print("It's a draw!")
    elif winner == GAME_BLACK_WIN:
        print("Black wins!")
    else:
        print("White wins!")