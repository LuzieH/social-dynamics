from typing import Callable
import numpy as np
import gin
from social_dynamics.agent_networks import agent_network


ActivationFunction = Callable[[np.ndarray], np.ndarray]


@gin.configurable()
def activation_function_builder(a: float, b: float):
    
    def activation_function(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-a * (x + b)))
    
    return activation_function


@gin.configurable()
class GODAgentNetwork(agent_network.AgentNetwork):
    
    def __init__(self, adjecency_matrix: np.ndarray, agents: np.ndarray,
                resistance: np.ndarray, attention: np.ndarray, inputs: np.ndarray,
                S1: ActivationFunction, S2: ActivationFunction) -> None:
        super().__init__(adjecency_matrix, agents)
        self._resistance = resistance
        self._attention = attention
        self._input = inputs
        self._S1 = S1
        self._S2 = S2
    
    def _step(self) -> None:
        """
        Updates the General Opinion Dynamics Agent Network according to the model
        presented in https://arxiv.org/abs/2009.04332
        """
        F = 

