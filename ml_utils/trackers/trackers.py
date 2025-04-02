from accelerate.tracking import GeneralTracker, on_main_process
import os
import yaml
import csv
import warnings
from aim import (
    Run as AimRun,
    Text as AimText,
    Image as AimImage,
    Distribution as AimDistribution,
    Figure as AimFigure,
)

from typing import Optional, Union, Dict, Any, Callable

from .utils import (
    generate_id,
    validate_experiment_name,
    update_yaml,
    compare_fns,
)


__all__ = [
    "AimTracker",
    "SimpleTracker",
    "SimpleGeneralTracker",
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
    def log_text(
        self,
        values: dict,
        step: Optional[int] = None,
        kwargs: Optional[Dict[str, dict]] = None,
    ):
        """
        Logs `text` to the current run.

        Args:
            values (`Dict[str, str]`):
                Values to be logged as key-value pairs. The values need to have type `str`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs (`Dict[str, dict]`):
                Additional key word arguments passed along to the `Run.Text` and `Run.track` method specified by the
                keys `aim_text` and `track`, respectively.
        """

        for key, value in values.items():
            self.writer.track(AimText(value), name=key, step=step, **kwargs)

    @on_main_process
    def finish(self):
        """
        Closes `aim` writer
        """
        self.writer.close()


class SimpleTracker:
    """
    A simple tracker class that can be used to track metrics and hyperparameters.
    """

    def __init__(
        self,
        log_path: str = "./.logs/",
    ):
        self.log_path = log_path
        self.experiment_path = None
        self.experiment_id = None
        self.experiment_name = None

    def is_experiment(self) -> bool:
        return os.path.isdir(self.experiment_path)

    def start_experiment(
        self,
        experiment_name: Optional[str] = None,
    ) -> None:
        assert validate_experiment_name(
            experiment_name
        ), f"Invalid experiment name: {experiment_name}"
        self.experiment_name = experiment_name

        experiment_id = generate_id()

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        experiment_path = os.path.join(self.log_path, experiment_id)
        assert not os.path.isdir(
            experiment_path
        ), f"Experiment already exists: {experiment_path}"
        os.makedirs(experiment_path)
        self.experiment_id = experiment_id
        self.experiment_path = experiment_path

        with open(os.path.join(experiment_path, "meta.yaml"), "w") as f:
            yaml.dump(
                {
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_name,
                },
                f,
            )

    def resume_experiment(self, experiment_subpath: str) -> None:
        self.experiment_path = os.path.join(self.log_path, experiment_subpath)
        assert self.is_experiment()
        with open(os.path.join(self.experiment_path, "meta.yaml"), "r") as f:
            meta = yaml.safe_load(f)
        self.experiment_id = meta["experiment_id"]
        self.experiment_name = meta["experiment_name"]

    def log_param(self, key: str, value: Any):
        assert self.is_experiment()
        update_yaml(os.path.join(self.experiment_path, "params.yaml"), {key: value})

    def log_params(self, params: dict):
        assert self.is_experiment()
        update_yaml(os.path.join(self.experiment_path, "params.yaml"), params)

    def log_value(
        self,
        key: str,
        value: Any,
        step: Optional[int] = None,
        compare_fn: Callable = compare_fns.new,
    ):
        assert self.is_experiment()
        # Update the values csv file
        if not os.path.isdir(os.path.join(self.experiment_path, "values")):
            os.makedirs(os.path.join(self.experiment_path, "values"))
        file = os.path.join(self.experiment_path, "values", f"{key}.csv")
        if not os.path.isfile(file):
            with open(file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "value"])
        with open(file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([step, value])

        # Update the value.yaml file
        file = os.path.join(self.experiment_path, "values.yaml")
        if os.path.isfile(file):
            with open(file, "r") as f:
                values = yaml.safe_load(f)
        else:
            values = {}
        key_ = f"{"last" if compare_fn is compare_fns.new else "best"}_{key}"
        values[key_] = compare_fn(
            value,
            values.get(key, None),
        )
        with open(os.path.join(self.experiment_path, "values.yaml"), "w") as f:
            yaml.dump(values, f)

    def log_values(
        self,
        value_map: dict,
        step: Optional[int] = None,
        compare_fn: Callable = compare_fns.new,
    ):
        assert self.is_experiment()
        for key, value in value_map.items():
            self.log_value(
                key=key,
                value=value,
                step=step,
                compare_fn=compare_fn,
            )

    def __repr__(self):
        return f"Logger(log_path={self.log_path})"


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

    @on_main_process
    def log_text(
        self,
        values: dict,
        step: Optional[int] = None,
        kwargs: Optional[Dict[str, dict]] = None,
    ):
        """
        Logs `text` to the current run.

        Args:
            values (`Dict[str, str]`):
                Values to be logged as key-value pairs. The values need to have type `str`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs (`Dict[str, dict]`):
                Additional key word arguments passed along to the `Run.Text` and `Run.track` method specified by the
                keys `aim_text` and `track`, respectively.
        """
        self.writer.log_values(values, step=step)
