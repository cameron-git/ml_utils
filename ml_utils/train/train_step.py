import torch
import torch.nn as nn
from accelerate import Accelerator

__all__ = ["train_step"]


def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
):
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs["loss"]
    accelerator.backward(loss)
    optimizer.step()
    return outputs
