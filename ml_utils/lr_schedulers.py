import math
from functools import partial
import torch

__all__ = ["get_lr_scheduler"]


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_rate: float = 0.0,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_steps: int,
    warmup_steps: int = 0,
    scheduler_type: str = "cosine",
):
    if scheduler_type == "cosine":
        lr_lambda = partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
            num_cycles=0.5,
            min_lr_rate=0.1,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )
    else:
        raise ValueError(f"Unknown scheduler type {scheduler_type}")
    return lr_scheduler
