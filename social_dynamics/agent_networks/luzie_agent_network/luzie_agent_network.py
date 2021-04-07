from collections import defaultdict
import numpy as np
import gin
from social_dynamics.agent_networks import agent_network
from social_dynamics.agent_networks.luzie_agent_network.builders import AdjMatrixBuilder, AgentsBuilder, ParamsBuilder
from typing import Optional


@gin.configurable()
class LuzieAgentNetwork(agent_network.AgentNetwork):

    def __init__(self,
                 adj_matrix_builder: AdjMatrixBuilder,
                 agents_builder: AgentsBuilder,
                 parameters_builder: ParamsBuilder,
                 time_interval: float = 0.01,
                 noise_std: float = 0,
                 builders_kwargs: Optional[dict] = None) -> None:
        """
        Implementation of Luzie's variation of the model presented in https://arxiv.org/abs/2009.04332.
        
        Args:
            adj_matrix_builder: Builds and returns the adjacency to be used by the
                        network object. This will define the network structure.
                        The adjacency matrix will have shape (n_agents, n_agents)
            agents_builder: Builds and returns the initial state of all the agents.
                        The agents' representation will have shape (n_agents, n_options)
            parameters_builder: Builds all the other parameters required to update the model.
                        Returns a dictionary containing all the necessary parameters.
            time_interval: time step for the Euler method applied at every step() call.
            noise_std: Standard deviation of the noise added to the computation of F 
                        upon step() calls.
        """
        if builders_kwargs is None:
            builders_kwargs = defaultdict(dict)
        # Parameters for the builders are usually passed via Gin Config
        adjacency_matrix = adj_matrix_builder(**builders_kwargs["adj_matrix_builder_kwargs"])
        agents = agents_builder(**builders_kwargs["agents_builder_kwargs"])
        params = parameters_builder(adjacency_matrix=adjacency_matrix, **builders_kwargs["parameters_builder_kwargs"])

        self._adjacency_tensor = params["adjacency_tensor"]
        self._d = params["d"]
        self._u = params["u"]
        self._v = params["v"]
        self._b = params["b"]
        self._S = np.tanh
        super().__init__(adjacency_matrix, agents)
        self._n_agents, self._n_options = self.agents.shape
        self._time_interval = time_interval
        self._noise_std = noise_std
        self._non_diag_bool_tensor = np.ones(shape=(self._n_agents, self._n_options, self._n_options),
                                             dtype=np.bool)
        for option in range(self._n_options):
            self._non_diag_bool_tensor[:, option, option] = False

    def _step(self, time_interval: float = None) -> None:
        """
        Updates the Agent Network. Euler method is used, with possibly added noise at
        every time step.
        
        If time_interval arg is provided, it will use this instead of the object's self._time_interval attribute.
        """
        t = time_interval or self._time_interval
        F = ((-self._d * self._agents
              + self._u * self._S(np.einsum('ikj,kj->ij', np.einsum('...ii->...i', self._adjacency_tensor),
                                            self._agents))
              + self._v * np.sum(self._S(np.einsum('ikjl,kl->ijl', self._adjacency_tensor, self._agents)),
                                 axis=2, where=self._non_diag_bool_tensor)
              + self._b
              ) * t
             + np.random.normal(size=(self._n_agents, self._n_options), scale=self._noise_std) * (t**0.5))

        delta_z = F - np.mean(F, axis=1, keepdims=True)

        self._agents += delta_z
