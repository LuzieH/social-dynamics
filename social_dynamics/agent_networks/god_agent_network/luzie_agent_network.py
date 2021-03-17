import numpy as np
import gin
from social_dynamics.agent_networks import agent_network
from social_dynamics.agent_networks.god_agent_network.builders import ActivationFunction


@gin.configurable()
class LuzieAgentNetwork(agent_network.AgentNetwork):

    def __init__(self,
                 adjacency_matrix: np.ndarray,
                 agents: np.ndarray,
                 adjacency_tensor: np.ndarray,
                 resistance: np.ndarray,
                 attention: np.ndarray,
                 inputs: np.ndarray,
                 time_interval: float = 0.01,
                 noise_std: float = 0) -> None:
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
            time_interval: time step for the Euler method applied at every step() call.
            noise_std: Standard deviation of the noise added to the computation of F 
                        upon step() calls.
        """
        self._n_agents, self._n_options = agents.shape
        super().__init__(adjacency_matrix, agents)
        self._adjacency_tensor = adjacency_tensor
        self._resistance = resistance
        self._attention = attention
        self._input = inputs
        self._S = np.tanh
        self._time_interval = time_interval
        self._noise_std = noise_std
        self._non_diag_bool_tensor = np.ones(shape=(self._n_agents, self._n_options, self._n_options),
                                             dtype=np.bool)
        for option in range(self._n_options):
            self._non_diag_bool_tensor[:, option, option] = False

    def _step(self, time_interval: float = None) -> None:
        """
        Updates the General Opinion Dynamics Agent Network according to the model presented in
        https://arxiv.org/abs/2009.04332 without enforcing the requirement that the average opinion of
        an agent be 0. Euler method is used, with possibly added noise at every time step.
        
        If time_interval arg is provided, it will use this instead of the object's self._time_interval attribute.
        """
        t = time_interval or self._time_interval
        F = (
            (-self._resistance * self._agents + self._attention *
             (self._S(np.einsum('ikj,kj->ij', np.einsum('...ii->...i', self._adjacency_tensor), self._agents))
              + np.sum(self._S(np.einsum('ikjl,kl->ijl', self._adjacency_tensor, self._agents)),
                       axis=2,
                       where=self._non_diag_bool_tensor)) + self._input) * t +
            np.random.normal(size=(self._n_agents, self._n_options), scale=self._noise_std) * (t**0.5))

        self._agents += F
