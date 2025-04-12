import torch
import torch.nn as nn
from accelerate import Accelerator
import transformers
from tqdm.auto import trange
from typing import Callable

from .train_step import train_step as mu_train_step
from ..metrics import log_metrics
from ..eval.test import test
from ..generation import log_generation

__all__ = ["train"]


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    tokenizer: transformers.PreTrainedTokenizerBase,
    accelerator: Accelerator,
    val_loader: torch.utils.data.DataLoader | None = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    train_step: Callable = mu_train_step,
    **kwargs,
):

    # Accelerate
    model, optimizer, train_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
    )
    if lr_scheduler is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    # Train
    train_steps = kwargs["train_steps"]
    log_interval = kwargs["log_interval"]
    val_interval = kwargs["val_interval"]
    save_interval = kwargs["save_interval"]
    max_val_steps = kwargs["max_val_steps"]
    metric_names = kwargs["metric_names"]
    max_grad_norm = kwargs.get("max_grad_norm", None)
    max_grad_value = kwargs.get("max_grad_value", None)
    seq_len = kwargs["seq_len"]

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
                lr_scheduler=lr_scheduler,
                max_grad_norm=max_grad_norm,
                max_grad_value=max_grad_value,
            )

            # Train metrics
            for k in train_metrics.keys():
                train_metrics[k] += outputs[
                    k
                ].item()  # TODO: Might need to use accelerator.gather

            # Train logging
            if log_interval is not None and step % log_interval == 0:
                train_metrics = log_metrics(
                    metrics=train_metrics,
                    step=step,
                    split="train",
                    log_interval=log_interval if step > 0 else 1,
                    accelerator=accelerator,
                    progress=progress,
                )
                accelerator.log(
                    {
                        "train_grad_norm": (
                            outputs["grad_norm"].item()
                            if outputs["grad_norm"] is not None
                            else None
                        )
                    },
                    step=step,
                )

            # Validation
            if val_interval is not None and step % val_interval == 0:
                # Validation metrics
                if val_loader is not None:
                    test(
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
                log_generation(
                    model=model,
                    tokenizer=tokenizer,
                    max_length=seq_len,
                    step=step,
                    accelerator=accelerator,
                )

                model.train()

            # Checkpoint
            if save_interval is not None and step % save_interval == 0 and step > 0:
                accelerator.wait_for_everyone()
                accelerator.save_state(
                    output_dir=f"{accelerator.project_dir}/checkpoints/{accelerator.start_time}/checkpoint_{step}",
                )

            # Step
            step += 1
            progress.update(1)
            progress.set_description("Training")
            if train_steps <= step:
                break

    progress.close()
