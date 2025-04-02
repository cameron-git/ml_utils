import os
from .utils import generate_id, validate_experiment_name, update_yaml, compare_fns
import yaml
import csv
from typing import Any, Callable, Optional


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
