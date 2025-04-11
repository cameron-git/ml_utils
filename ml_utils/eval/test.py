import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from tqdm.auto import trange

from ..metrics import log_metrics

__all__ = ["test"]


def test(
    model: nn.Module,
    split: str,
    test_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    progress: tqdm,
    metric_names: list[str],
    step: int | None = None,
    max_test_steps: int | None = None,
):
    test_loader = accelerator.prepare(test_loader)
    test_progress = False
    print(max_test_steps)
    if progress is None:
        test_progress = True
        try:
            stop = max_test_steps if max_test_steps is not None else len(test_loader)
        except:
            stop = 0
        progress = trange(
            stop,
            disable=not accelerator.is_local_main_process,
        )
    progress.set_description(f"Testing {split}")
    model.eval()
    metrics = {k: 0 for k in metric_names}
    with torch.no_grad():
        # Test step
        for test_step, batch in enumerate(test_loader):
            outputs = model(**batch)
            for k in metrics.keys():
                metrics[k] += outputs[k].item()
            if test_progress:
                progress.update(1)
                progress.set_postfix(
                    {k: v / (test_step + 1) for k, v in metrics.items()}
                )
            if max_test_steps is not None and test_step >= max_test_steps:
                break

    log_metrics(
        metrics=metrics,
        step=step,
        split=split,
        log_interval=test_step + 1,
        accelerator=accelerator,
        progress=progress,
    )
