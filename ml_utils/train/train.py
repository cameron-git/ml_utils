import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm.auto import trange
from typing import Callable

import ml_utils as mu

__all__ = ["train"]


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None = None,
    test_loader: torch.utils.data.DataLoader | None = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    train_step: Callable = mu.train.train_step,
    **kwargs,
):

    # Accelerate

    accelerator = Accelerator(
        mixed_precision=kwargs["mixed_precision"],
        log_with=[mu.trackers.AimTracker, mu.trackers.SimpleGeneralTracker],
        gradient_accumulation_steps=kwargs["gradient_accumulation_steps"],
        cpu=kwargs["use_cpu"],
        # rng_types=None, # TODO: Implement RNG types
        project_dir=kwargs["project_dir"],
    )
    model, optimizer, train_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
    )
    if lr_scheduler is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if test_loader is not None:
        test_loader = accelerator.prepare(test_loader)
    accelerator.init_trackers(
        "llm",
        kwargs,
    )

    # Train
    train_steps = kwargs["train_steps"]
    log_interval = kwargs["log_interval"]
    val_interval = kwargs["val_interval"]
    max_val_steps = kwargs["max_val_steps"]
    metric_names = kwargs["metric_names"]

    train_metrics = {k: 0 for k in metric_names}
    model.train()
    step = 0
    progress = trange(
        train_steps,
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )
    while step < train_steps:
        for batch in train_loader:
            # Train step
            outputs = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                accelerator=accelerator,
            )

            # Train metrics
            for k, v in train_metrics.items():
                v += outputs[k].item()  # TODO: Might need to use accelerator.gather

            # Train logging
            if step % log_interval == 0:
                train_metrics = mu.log_metrics(
                    metrics=train_metrics,
                    step=step,
                    split="train",
                    log_interval=log_interval,
                    accelerator=accelerator,
                    progress=progress,
                )

            # Validation
            if val_loader is not None and step % val_interval == 0:
                mu.eval.test(
                    model=model,
                    split="val",
                    test_loader=val_loader,
                    accelerator=accelerator,
                    progress=progress,
                    metric_names=metric_names,
                    step=step,
                    max_test_steps=max_val_steps,
                )

                # Generate

                # TODO: Implement generation

                # Checkpoint

                # TODO: Implement checkpointing

                model.train()

            # Step
            step += 1
            progress.update(1)
            progress.set_description("Training")
            if train_steps <= step:
                break

    progress.close()
