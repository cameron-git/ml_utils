import math
from functools import partial
from typing import Callable
import torch

__all__ = ["get_lr_scheduler"]


def cosine_lr_lambda(
    current_step: int,
    *,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    num_cycles: float = 0.5,
    min_lr_rate: float = 0.1,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def linear_lr_lambda(
    current_step: int,
    *,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    min_lr_rate: float = 0.0,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    factor = 1.0 - progress
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def constant_2_linear_lr_lambda(
    current_step: int,
    *,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    min_lr_rate: float = 0.0,
    phase_ratio: float = 0.8,
    num_2nd_warmup_steps: int = 0,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_training_steps * phase_ratio:
        return 1.0
    if current_step < num_training_steps * phase_ratio + num_2nd_warmup_steps:
        return float(current_step - num_training_steps * phase_ratio) / float(
            max(1, num_2nd_warmup_steps)
        )
    progress = float(
        current_step - num_training_steps * phase_ratio - num_2nd_warmup_steps
    ) / float(
        max(
            1,
            num_training_steps
            - num_training_steps * phase_ratio
            - num_2nd_warmup_steps,
        )
    )
    factor = 1.0 - progress
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    lr_lambda: Callable = None,
    **kwargs,
):
    """
    scheduler_type: str, one of ["lambda", "cosine", "linear"]
    kwargs:
        num_training_steps: int,
        num_warmup_steps: int,
        min_lr_rate: float,
        phase_ratio: float,
        num_2nd_warmup_steps: int,
    """
    if scheduler_type == "lambda":
        assert lr_lambda is not None, "lr_lambda must be provided for lambda scheduler"
        lr_lambda = lr_lambda
    elif scheduler_type == "cosine":
        lr_lambda = partial(
            cosine_lr_lambda,
            **kwargs,
        )
    elif scheduler_type == "linear":
        lr_lambda = partial(
            linear_lr_lambda,
            **kwargs,
        )
    elif scheduler_type == "constant_2_linear":
        lr_lambda = partial(
            constant_2_linear_lr_lambda,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown scheduler type {scheduler_type}")
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )
    return lr_scheduler
