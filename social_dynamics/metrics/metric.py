from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class Metric(ABC):
    
    def __init__(self, name) -> None:
        self._name = name
    
    @property
    def name(self):
        return self._name
    
    @abstractmethod
    def call(self, *args, **kwargs) -> None:
        """Accumulates statistics for the metric. Users should use __call__ instead."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Resets the values being tracked by the metric."""
        pass
    
    @abstractmethod
    def result(self) -> Union[np.ndarray, float]:
        """Computes and returns a final value for the metric."""
        pass
    
    def __call__(self, *args, **kwargs) -> None:
        return self.call(*args, **kwargs)


