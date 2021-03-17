from abc import ABC, abstractmethod
import numpy as np


class AgentNetwork(ABC):

    def __init__(self, adjacency_matrix: np.ndarray, agents: np.ndarray) -> None:
        self._adjacency_matrix = adjacency_matrix
        self._agents = agents

    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix

    @property
    def agents(self):
        return self._agents

    def step(self, **kwargs) -> None:
        self._step(**kwargs)

    @abstractmethod
    def _step(self) -> None:
        pass
