from accelerate.tracking import GeneralTracker, on_main_process
import os
import warnings
from aim import (
    Run as AimRun,
    Text as AimText,
    Image as AimImage,
    Distribution as AimDistribution,
    Figure as AimFigure,
)

from typing import Optional, Union, Dict, Any, List

from .simple import SimpleTracker


__all__ = [
    "AimTracker",
    "SimpleGeneralTracker",
    "WandBTracker",
]


class AimTracker(GeneralTracker):
    """
    A `Tracker` class that supports `aim`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run.
        **kwargs (additional keyword arguments, *optional*):
            Additional key word arguments passed along to the `Run.__init__` method.
    """

    name = "aim"
    requires_logging_directory = False

    @on_main_process
    def __init__(
        self,
        run_name: str,
        logging_dir: Optional[Union[str, os.PathLike]] = ".",
        **kwargs,
    ):
        self.run_name = run_name

        self.writer = AimRun(repo=logging_dir, **kwargs)
        self.writer.name = self.run_name

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (`dict`):
                Values to be stored as initial hyperparameters as key-value pairs.
        """
        self.writer["hparams"] = values

    @on_main_process
    def log(self, values: dict, step: Optional[int], **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (`dict`):
                Values to be logged as key-value pairs.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `Run.track` method.
        """
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = AimText(value)
            self.writer.track(value, name=key, step=step, **kwargs)

    @on_main_process
    def log_images(
        self,
        values: dict,
        step: Optional[int] = None,
        kwargs: Optional[Dict[str, dict]] = None,
    ):
        """
        Logs `images` to the current run.

        Args:
            values (`Dict[str, Union[np.ndarray, PIL.Image, Tuple[np.ndarray, str], Tuple[PIL.Image, str]]]`):
                Values to be logged as key-value pairs. The values need to have type `np.ndarray` or PIL.Image. If a
                tuple is provided, the first element should be the image and the second element should be the caption.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs (`Dict[str, dict]`):
                Additional key word arguments passed along to the `Run.Image` and `Run.track` method specified by the
                keys `aim_image` and `track`, respectively.
        """
        aim_image_kw = {}
        track_kw = {}

        if kwargs is not None:
            aim_image_kw = kwargs.get("aim_image", {})
            track_kw = kwargs.get("track", {})

        for key, value in values.items():
            if isinstance(value, tuple):
                img, caption = value
            else:
                img, caption = value, ""
            aim_image = AimImage(img, caption=caption, **aim_image_kw)
            self.writer.track(aim_image, name=key, step=step, **track_kw)

    @on_main_process
    def finish(self):
        """
        Closes `aim` writer
        """
        self.writer.close()


class SimpleGeneralTracker(GeneralTracker):
    """
    A simple tracker class that can be used to track metrics and hyperparameters.
    """

    name = "simple"
    requires_logging_directory = False

    @on_main_process
    def __init__(
        self,
        run_name: str,
        logging_dir: Optional[Union[str, os.PathLike]] = ".",
        **kwargs,
    ):
        self.run_name = run_name

        self.writer = SimpleTracker(os.path.join(logging_dir, ".logs"))
        self.writer.start_experiment(run_name)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (`dict`):
                Values to be stored as initial hyperparameters as key-value pairs.
        """
        self.writer.log_params(values)

    @on_main_process
    def log(self, values: dict, step: Optional[int], **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (`dict`):
                Values to be logged as key-value pairs.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `Run.track` method.
        """
        self.writer.log_values(values, step=step)

    @on_main_process
    def log_images(
        self,
        values: dict,
        step: Optional[int] = None,
        kwargs: Optional[Dict[str, dict]] = None,
    ):
        """
        Logs `images` to the current run.

        Args:
            values (`Dict[str, Union[np.ndarray, PIL.Image, Tuple[np.ndarray, str], Tuple[PIL.Image, str]]]`):
                Values to be logged as key-value pairs. The values need to have type `np.ndarray` or PIL.Image. If a
                tuple is provided, the first element should be the image and the second element should be the caption.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs (`Dict[str, dict]`):
                Additional key word arguments passed along to the `Run.Image` and `Run.track` method specified by the
                keys `aim_image` and `track`, respectively.
        """
        warnings.warn(
            "SimpleTracker does not support logging images. Skipping.",
        )


class WandBTracker(GeneralTracker):
    """
    A `Tracker` class that supports `wandb`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run.
        **kwargs (additional keyword arguments, *optional*):
            Additional key word arguments passed along to the `wandb.init` method.
    """

    name = "wandb"
    requires_logging_directory = False
    main_process_only = False

    @on_main_process
    def __init__(
        self,
        project_name: str,
        run_name: str,
        **kwargs,
    ):
        super().__init__()
        self.run_name = run_name

        import wandb

        self.run = wandb.init(project=project_name, name=self.run_name, **kwargs)

    @property
    def tracker(self):
        return self.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        import wandb

        wandb.config.update(values, allow_val_change=True)

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `wandb.log` method.
        """
        import wandb

        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = wandb.Table(data=[[value]], columns=["Text"])
            if isinstance(value, dict):
                value = wandb.Table(
                    data=[[v for v in value.values()]], columns=list(value.keys())
                )
            self.run.log({key: value}, step=step, **kwargs)

    @on_main_process
    def log_images(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `images` to the current run.

        Args:
            values (Dictionary `str` to `List` of `np.ndarray` or `PIL.Image`):
                Values to be logged as key-value pairs. The values need to have type `List` of `np.ndarray` or
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `wandb.log` method.
        """
        import wandb

        for k, v in values.items():
            self.log({k: [wandb.Image(image) for image in v]}, step=step, **kwargs)

    @on_main_process
    def log_table(
        self,
        table_name: str,
        columns: List[str] = None,
        data: List[List[Any]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
        **kwargs,
    ):
        """
        Log a Table containing any object type (text, image, audio, video, molecule, html, etc). Can be defined either
        with `columns` and `data` or with `dataframe`.

        Args:
            table_name (`str`):
                The name to give to the logged table on the wandb workspace
            columns (list of `str`, *optional*):
                The name of the columns on the table
            data (List of List of Any data type, *optional*):
                The data to be logged in the table
            dataframe (Any data type, *optional*):
                The data to be logged in the table
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        import wandb

        values = {
            table_name: wandb.Table(columns=columns, data=data, dataframe=dataframe)
        }
        self.log(values, step=step, **kwargs)

    @on_main_process
    def finish(self):
        """
        Closes `wandb` writer
        """
        self.run.finish()
