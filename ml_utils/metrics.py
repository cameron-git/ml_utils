from accelerate import Accelerator
from tqdm import tqdm

__all__ = [
    "log_metrics",
]


def log_metrics(
    metrics: dict[str, float | int],
    step: int,
    split: str,
    log_interval: int,
    accelerator: Accelerator,
    progress: tqdm | None = None,
):
    metrics = {f"{split}_k": v / log_interval for k, v in metrics.items()}
    if progress is not None:
        progress.set_postfix(**metrics)
    accelerator.log(metrics, step=step)
    new_metrics = {k: 0 for k in metrics.keys()}
    return new_metrics
