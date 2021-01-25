from typing import Callable
import numpy as np
import gin
from social_dynamics.agent_networks import agent_network


ActivationFunction = Callable[[np.ndarray], np.ndarray]


@gin.configurable()
def activation_function_builder(a: float, b: float, c: float):
    
    def activation_function(x: np.ndarray) -> np.ndarray:
        return a*np.tanh(b*x + c*np.tanh(x**2))
    
    return activation_function


@gin.configurable()
class GODAgentNetwork(agent_network.AgentNetwork):
    
    def __init__(self, adjacency_matrix: np.ndarray, agents: np.ndarray,
                resistance: np.ndarray, attention: np.ndarray, inputs: np.ndarray,
                S1: ActivationFunction, S2: ActivationFunction, 
                time_interval: float = 0.01) -> None:
        """
        Args:
            adjacency_matrix: Defines the network structure and the influence params.
                        Shape = (n_agents, n_agents, n_opinions, n_opinions)
            agents: States for each agent. Shape = (n_agents, n_opinions)
            resistance: Matrix of resistance params for every agent and opinion.
                        Shape = (n_agents, n_opinions)
            attention: Vector of attention params for every agent. Shape = (n_agents, 1)
            inputs: Matrix of input params for every agent and opinion.
                        Shape=(n_agents, n_opinions)
            S1: First activation function for intra-opinion activations
            S2: Second activation functions for inter-opinion activations.
            time_interval: time step for the Euler method applied at every step() call.
        """
        self._n_agents, self._n_options = agents.shape
        super().__init__(adjacency_matrix, agents)
        self._resistance = resistance
        self._attention = attention
        self._input = inputs
        self._S1 = S1
        self._S2 = S2
        self._time_interval = time_interval
        self._non_diag_bool_tensor = np.ones(shape=(self._n_agents, self._n_options, self._n_options), dtype=np.bool)
        for option in range(self._n_options): 
            self._non_diag_bool_tensor[:, option, option] = False
    
    def _step(self, time_interval: float = None) -> None:
        """
        Updates the General Opinion Dynamics Agent Network according to the model
        presented in https://arxiv.org/abs/2009.04332. Euler method is used.
        
        If time_interval arg is provided, it will use this instead of the object's self._time_interval attribute.
        """
        F = (-self._resistance*self._agents + 
             self._attention*(self._S1(np.einsum('ikj,kj->ij', np.einsum('...ii->...i', self._adjacency_matrix),
                                                 self._agents)) +
                              np.sum(self._S2(np.einsum('ikjl,kl->ijl', self._adjacency_matrix, self._agents)),
                                     axis=2, where=self._non_diag_bool_tensor)) + 
             self._input)
        
        if time_interval:
            self._agents += time_interval*F
        else:
            self._agents += self._time_interval*F

