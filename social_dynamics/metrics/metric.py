from abc import ABC, abstractmethod
from typing import Dict, Union
import numpy as np


class Metric(ABC):

    def __init__(self, name: str, buffer_size: int) -> None:
        self._name = name
        self._buffer_size = buffer_size

    @property
    def name(self) -> str:
        return self._name
    
    def __call__(self, *args, **kwargs) -> None:
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs) -> None:
        """Accumulates statistics for the metric. Users should use __call__ instead."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the values being tracked by the metric."""

    @abstractmethod
    def result(self) -> Union[Dict[str, np.ndarray], float, np.ndarray]:
        """Computes and returns a final value for the metric."""

    @abstractmethod
    def save(self, save_path: str, time_step: int) -> None:
        """Saves the cumulated results of the metric

        Args:
            save_path (str): Path to the experiment_dir where to create the metric's save
                        folder and save its results
            time_step (int): time step associated with current save call
        """
    
