from typing import Tuple, Dict
import gin
import numpy as np
import os
from scipy.spatial.distance import pdist
from social_dynamics.agent_networks.agent_network import AgentNetwork
from social_dynamics.metrics import metric


@gin.configurable()
class StateMetric(metric.Metric):

    def __init__(self, buffer_size: int, shape: Tuple[int], name: str = 'StateMetric') -> None:
        super().__init__(name=name, buffer_size=buffer_size)
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
    
    def save(self, save_path: str, time_step: int) -> None:
        metric_dir = os.path.join(save_path, self.name)
        if not os.path.exists(metric_dir):
            os.mkdir(metric_dir)
        
        np.save(os.path.join(metric_dir, 'results_t{}.npy'.format(time_step)),
                self.result())


@gin.configurable()
class PopulationStateMetric(metric.Metric):

    def __init__(self, buffer_size: int, n_agents: int, n_options: int, theta: float, name: str = 'PopulationStateMetric') -> None:
        """
        This metric expects to be called on AgentNetwork objects whose agents attribute
        has shape (n_agents, n_options)
        """
        super().__init__(name=name, buffer_size=buffer_size)
        self._buffer_size = buffer_size
        self._agreement_buffer = np.zeros(shape=(buffer_size, n_options))
        self._consensus_buffer = np.zeros(shape=(buffer_size,))
        self._dissensus_buffer = np.zeros(shape=(buffer_size,))
        self._unopinionated_buffer = np.zeros(shape=(buffer_size, n_agents))
        self._theta = theta
        self._cursor = 0

    def call(self, agent_network: AgentNetwork) -> None:
        likes = agent_network.agents > self._theta
        dislikes = agent_network.agents < -self._theta

        agreement = np.logical_and(np.all(likes == likes[0], axis=0), np.all(dislikes == dislikes[0], axis=0))
        consensus = np.all(agreement) and np.all(pdist(agent_network.agents) <= self._theta)
        dissensus = np.all(np.mean(agent_network.agents, axis=0) <= self._theta)
        unopinionated = np.all(np.logical_and(~likes, ~dislikes), axis=1)
        
        
        self._agreement_buffer[self._cursor] = agreement
        self._consensus_buffer[self._cursor] = consensus
        self._dissensus_buffer[self._cursor] = dissensus
        self._unopinionated_buffer[self._cursor] = unopinionated
        
        self._cursor = (self._cursor + 1) % self._buffer_size

    def reset(self) -> None:
        self._agreement_buffer.fill(0.0)
        self._consensus_buffer.fill(0.0)
        self._dissensus_buffer.fill(0.0)
        self._unopinionated_buffer.fill(0.0)
        
        self._cursor = 0

    def result(self) -> Dict[str, np.ndarray]:
        return {"AgreementMetric": np.copy(self._agreement_buffer),
                "ConsensusMetric": np.copy(self._consensus_buffer),
                "DissensusMetric": np.copy(self._dissensus_buffer),
                "UnopinionatedMetric": np.copy(self._unopinionated_buffer)}
    
    def save(self, save_path: str, time_step: int) -> None:
        results = self.result()
        for metric in results:
            metric_dir = os.path.join(save_path, metric)
            if not os.path.exists(metric_dir):
                os.mkdir(metric_dir)
        
            np.save(os.path.join(metric_dir, 'results_t{}.npy'.format(time_step)),
                    results[metric])
