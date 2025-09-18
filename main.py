from toguzkumalak import run_game_vs_ai, run_game
from toguzkumalak.train import *
from toguzkumalak.eval_vs_minimax import play_match

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_selfplay(model_path="dqn_togyz.pt", save_path="dqn_togyz_v2.pt", snapshot_every=500)
    model = DQN().to(device)
    model.load_state_dict(torch.load("dqn_togyz_v2.pt", map_location="cpu"))
    play_match(model, games=50, dqn_as_white=True, max_depth=6, time_limit_s=2.0, render_every=1)