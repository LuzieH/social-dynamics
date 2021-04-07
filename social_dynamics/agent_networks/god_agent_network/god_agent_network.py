from collections import defaultdict
import numpy as np
import gin
from social_dynamics.agent_networks import agent_network
from social_dynamics.agent_networks.god_agent_network.builders import ActivationFunction, AdjMatrixBuilder, AgentsBuilder, ParamsBuilder


@gin.configurable()
class GODAgentNetwork(agent_network.AgentNetwork):

    def __init__(self,
                 adj_matrix_builder: AdjMatrixBuilder,
                 agents_builder: AgentsBuilder,
                 parameters_builder: ParamsBuilder,
                 S1: ActivationFunction,
                 S2: ActivationFunction,
                 time_interval: float = 0.01,
                 noise_std: float = 0,
                 builders_kwargs: dict = defaultdict(dict)) -> None:
        """
        Implementation of the model presented in https://arxiv.org/abs/2009.04332.
        
        Args:
            adj_matrix_builder: Builds and returns the adjacency to be used by the
                        network object. This will define the network structure.
                        The adjacency matrix will have shape (n_agents, n_agents)
            agents_builder: Builds and returns the initial state of all the agents.
                        The agents' representation will have shape (n_agents, n_options)
            parameters_builder: Builds all the other parameters required to update the model.
                        Returns a dictionary containing all the necessary parameters.
            S1: First activation function for intra-opinion activations
            S2: Second activation functions for inter-opinion activations.
            time_interval: time step for the Euler method applied at every step() call.
            noise_std: Standard deviation of the noise added to the computation of F 
                        upon step() calls.
        """
        # Parameters for the builders are usually passed via Gin Config
        adjacency_matrix = adj_matrix_builder(**builders_kwargs["adj_matrix_builder_kwargs"])
        agents = agents_builder(**builders_kwargs["agents_builder_kwargs"])
        params = parameters_builder(adjacency_matrix=adjacency_matrix, **builders_kwargs["parameters_builder_kwargs"])

        self._adjacency_tensor = params["adjacency_tensor"]
        self._d = params["d"]
        self._u = params["u"]
        self._b = params["b"]
        self._S1 = S1
        self._S2 = S2
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
        Updates the General Opinion Dynamics Agent Network. Euler method is used, with possibly added noise at
        every time step.
        
        If time_interval arg is provided, it will use this instead of the object's self._time_interval attribute.
        """
        t = time_interval or self._time_interval
        F = (
            (-self._d * self._agents
             + self._u * (self._S1(np.einsum('ikj,kj->ij', np.einsum('...ii->...i', self._adjacency_tensor),
                                             self._agents))
                          + np.sum(self._S2(np.einsum('ikjl,kl->ijl', self._adjacency_tensor, self._agents)),
                                   axis=2, where=self._non_diag_bool_tensor)
                          )
             + self._b
            ) * t
            + np.random.normal(size=(self._n_agents, self._n_options), scale=self._noise_std) * (t**0.5))

        delta_z = F - np.mean(F, axis=1, keepdims=True)

        self._agents += delta_z
