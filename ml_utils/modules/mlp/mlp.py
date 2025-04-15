import torch
import torch.nn as nn
from typing import Callable


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.activation = activation

        self.fc1 = nn.Linear(embed_dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(embed_dim, mlp_dim, bias=False)
        self.fc3 = nn.Linear(mlp_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.activation(self.fc1(x))
        x2 = self.fc2(x)
        x = x1 * x2
        x = self.fc3(x)
        return x
