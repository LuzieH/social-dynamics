from typing import Callable
import numpy as np
import gin


ActivationFunction = Callable[[np.ndarray], np.ndarray]


@gin.configurable()
def activation_function_builder(a: float, b: float, c: float) -> ActivationFunction:
    
    def activation_function(x: np.ndarray) -> np.ndarray:
        return a*np.tanh(b*x + c*np.tanh(x**2))
    
    return activation_function


@gin.configurable()
def complete_adjacency_matrix_builder(n_agents: int):
    adjacency_matrix = np.ones(shape=(n_agents, n_agents))
    return adjacency_matrix


@gin.configurable()
def homogenous_adjacency_tensor_builder(adjacency_matrix: np.ndarray, n_options: int,
                             alpha: float, beta: float, gamma: float, delta: float):
    """
    Builds the adjacency tensor for the homogenous agents case presented in https://arxiv.org/abs/2009.04332.
    """
    n_agents = adjacency_matrix.shape[0]
    adjacency_tensor = np.ones(shape=(n_agents, n_agents, n_options, n_options))*delta
    option_diag = np.diagonal(adjacency_tensor, axis1=2, axis2=3)
    option_diag.setflags(write=1)       # np.diagonal return value is read-only view of input array
    option_diag[:, :, :] = gamma
    
    agent_diag = np.diagonal(adjacency_tensor, axis1=0, axis2=1)
    agent_diag.setflags(write=1)        # np.diagonal return value is read-only view of input array
    agent_diag[:, :, :] = beta
    
    both_diag = np.diagonal(option_diag, axis1=0, axis2=1)
    both_diag.setflags(write=1)         # np.diagonal return value is read-only view of input array
    both_diag[:, :] = alpha
    
    adjacency_tensor = np.einsum('ijkl,ij->ijkl', adjacency_tensor, adjacency_matrix)
    
    return adjacency_tensor


@gin.configurable()
def homogenous_agents_builder(n_agents: int, n_options: int):
    agents = np.random.normal(size=(n_agents, n_options))
    agents -= np.mean(agents, axis=1, keepdims=True)
    return agents


@gin.configurable()
def homogenous_resistance_builder(n_agents: int, n_options: int, d: float):
    resistance = np.ones(shape=(n_agents, n_options)) * d
    return resistance


@gin.configurable()
def homogenous_attention_builder(n_agents: int, u: float):
    attention = np.ones(shape=(n_agents, 1)) * u
    return attention


@gin.configurable()
def homogenous_inputs_builder(n_agents: int, n_options: int, b: float):
    inputs = np.ones(shape=(n_agents, n_options)) * b
    return inputs


