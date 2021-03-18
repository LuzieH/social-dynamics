import gin
import numpy as np
from social_dynamics.agent_networks.luzie_agent_network import agent_types
from typing import Callable, Dict, List

AdjMatrixBuilder = Callable[[int], np.ndarray]
AgentsBuilder = Callable[[int, int], np.ndarray]
ParamsBuilder = Callable[..., Dict[str, np.ndarray]]


@gin.configurable(module="Luzie")
def complete_adjacency_matrix_builder(n_agents: int) -> np.ndarray:
    """Builds an adjacency matrix of ones; corresponding to a complete network structure"""
    return np.ones(shape=(n_agents, n_agents))


@gin.configurable(module="Luzie")
def random_normal_agent_builder(n_agents: int, n_options: int) -> np.ndarray:
    """Initializes the states of all the agents using a Normal distribution."""
    return np.random.normal(size=(n_agents, n_options))


@gin.configurable(module="Luzie")
def homogenous_builder(adjacency_matrix: np.ndarray, n_options: int, alpha: float, beta: float, gamma: float,
                       delta: float, d: float, u: float, v: float, b: float) -> Dict[str, np.ndarray]:
    """
    Sets all the parameters and tensors to update a General Opinion Dynamics-style network.
    Follows the structure presented for the homogenous case in https://arxiv.org/abs/2009.04332.
    """
    # Adjacency Tensor
    n_agents = adjacency_matrix.shape[0]
    adjacency_tensor = np.ones(shape=(n_agents, n_agents, n_options, n_options)) * delta
    option_diag = np.diagonal(adjacency_tensor, axis1=2, axis2=3)
    option_diag.setflags(write=1)  # np.diagonal return value is read-only view of input array
    option_diag[:, :, :] = gamma
    agent_diag = np.diagonal(adjacency_tensor, axis1=0, axis2=1)
    agent_diag.setflags(write=1)  # np.diagonal return value is read-only view of input array
    agent_diag[:, :, :] = beta
    both_diag = np.diagonal(option_diag, axis1=0, axis2=1)
    both_diag.setflags(write=1)  # np.diagonal return value is read-only view of input array
    both_diag[:, :] = alpha
    adjacency_tensor = np.einsum('ijkl,ij->ijkl', adjacency_tensor, adjacency_matrix)

    # Update rule parameters
    resistance = np.ones(shape=(n_agents, n_options)) * d
    same_option_attention = np.ones(shape=(n_agents, 1)) * u
    other_options_attention = np.ones(shape=(n_agents, 1)) * v
    inputs = np.ones(shape=(n_agents, n_options)) * b

    params = {
        "adjacency_tensor": adjacency_tensor,
        "d": resistance,
        "u": same_option_attention,
        "v": other_options_attention,
        "b": inputs
    }

    return params

