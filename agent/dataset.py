import pickle
from typing import Sequence, Tuple
import torch
from torch.utils.data import Dataset

# x: [21], pi: [9], v: scalar
class BoardDataset(Dataset):
    def __init__(
        self,
        states: Sequence[torch.Tensor],
        policies: Sequence[torch.Tensor],
        values: Sequence[float] | Sequence[torch.Tensor],
        *,
        to_float32: bool = True,
    ):
        assert len(states) == len(policies) == len(values)
        
        self.x = [t.detach().cpu() for t in states]
        self.pi = [p.detach().cpu() for p in policies]
        self.v = [torch.as_tensor(v, dtype=torch.float32).view(1).cpu() for v in values]

        if to_float32:
            self.x = [t.to(torch.float32) for t in self.x]
            self.pi = [p.to(torch.float32) for p in self.pi]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.pi[idx], self.v[idx]

    @staticmethod
    def collate(batch):
        xs, pis, vs = zip(*batch)
        return (
            torch.stack(xs, dim=0),                 # [B,21]
            torch.stack(pis, dim=0),                # [B,9]
            torch.stack(vs, dim=0).squeeze(-1),     # [B]
        )

    def to_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "x": [t.numpy() for t in self.x],
                    "pi": [t.numpy() for t in self.pi],
                    "v": [float(t.item()) for t in self.v],
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def from_pickle(cls, path: str) -> "BoardDataset":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        states = [torch.from_numpy(a) for a in obj["x"]]
        policies = [torch.from_numpy(a) for a in obj["pi"]]
        values = obj["v"]
        return cls(states, policies, values)

    def to_pt(self, path: str):
        torch.save(
            {
                "x": torch.stack(self.x),
                "pi": torch.stack(self.pi),
                "v": torch.tensor([float(t.item()) for t in self.v], dtype=torch.float32),
            },
            path,
        )

    @classmethod
    def from_pt(cls, path: str) -> "BoardDataset":
        blob = torch.load(path, map_location="cpu")
        N = blob["x"].size(0)
        states = [blob["x"][i] for i in range(N)]
        policies = [blob["pi"][i] for i in range(N)]
        values = [blob["v"][i].item() for i in range(N)]
        return cls(states, policies, values)
