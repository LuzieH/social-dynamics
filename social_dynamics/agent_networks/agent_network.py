from abc import ABC, abstractmethod
import numpy as np


class AgentNetwork(ABC):
    
    def __init__(self, adjecency_matrix: np.ndarray, agents: np.ndarray) -> None:
        self._adjacency_matrix = adjecency_matrix
        self._agents = agents
    
    def step(self, **kwargs) -> None:
        self._step(self, **kwargs)
    
    @abstractmethod()
    def _step(self) -> None:
        pass

