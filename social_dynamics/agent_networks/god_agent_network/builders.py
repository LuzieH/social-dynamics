import gin
import numpy as np
from typing import Callable, Dict


ActivationFunction = Callable[[np.ndarray], np.ndarray]
AdjMatrixBuilder = Callable[[int], np.ndarray]
AgentsBuilder = Callable[[int, int], np.ndarray]
ParamsBuilder = Callable[..., Dict[str, np.ndarray]]


@gin.configurable()
def activation_function_builder(a: float, b: float, c: float) -> ActivationFunction:

    def activation_function(x: np.ndarray) -> np.ndarray:
        return a * np.tanh(b * x + c * np.tanh(x**2))

    return activation_function


@gin.configurable(module="GOD")
def complete_adjacency_matrix_builder(n_agents: int) -> np.ndarray:
    adjacency_matrix = np.ones(shape=(n_agents, n_agents))
    return adjacency_matrix


@gin.configurable(module="GOD")
def random_normal_agent_builder(n_agents: int, n_options: int) -> np.ndarray:
    agents = np.random.normal(size=(n_agents, n_options))
    agents -= np.mean(agents, axis=1, keepdims=True)
    return agents


@gin.configurable(module="GOD")
def homogenous_parameters_builder(adjacency_matrix: np.ndarray, n_options: int, alpha: float, beta: float, gamma: float,
                       delta: float, d: float, u: float, b: float) -> Dict[str, np.ndarray]:
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
    attention = np.ones(shape=(n_agents, 1)) * u
    inputs = np.ones(shape=(n_agents, n_options)) * b
    
    params = {"adjacency_tensor": adjacency_tensor,
              "resistance": resistance,
              "attention": attention,
              "inputs": inputs,
              }

    return params

