from typing import Tuple
import gin
import numpy as np
from social_dynamics.agent_networks.agent_network import AgentNetwork
from social_dynamics.metrics import metric


@gin.configurable()
class AgentMetric(metric.Metric):
    
    def __init__(self, buffer_size: int, shape: Tuple[int], name: str ='AgentMetric') -> None:
        super().__init__(name=name)
        self._buffer_size = buffer_size
        self._buffer = np.zeros(shape=(buffer_size, *shape))
        self._cursor = 0

    def call(self, agent_network: AgentNetwork) -> None:
        self._buffer[self._cursor] = agent_network.agents
        self._cursor = (self._cursor + 1) % self._buffer_size
    
    def reset(self) -> None:
        self._buffer.fill(0.0)
        self._cursor = 0
    
    def result(self) -> np.ndarray:
        return np.copy(self._buffer)

