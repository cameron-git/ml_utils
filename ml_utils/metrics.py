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
    keys = list(metrics.keys())
    metrics = {f"{split}_{k}": v / log_interval for k, v in metrics.items()}
    if progress is not None:
        if hasattr(progress, "postfix_dict"):
            progress.postfix_dict.update(metrics)
        else:
            progress.postfix_dict = metrics
        progress.set_postfix(**progress.postfix_dict)
    accelerator.log(metrics, step=step)
    new_metrics = {k: 0 for k in keys}
    return new_metrics
