import gin
import numpy as np
from typing import Callable, Dict, List, Union

AdjMatrixBuilder = Callable[..., np.ndarray]
AgentsBuilder = Callable[[int, int], np.ndarray]
ParamsBuilder = Callable[..., Dict[str, np.ndarray]]
AgentType = Dict[str, Union[List[float], float]]


@gin.configurable(module="Luzie")
def complete_adjacency_matrix_builder(n_agents: int) -> np.ndarray:
    """Builds an adjacency matrix of ones; corresponding to a complete network structure"""
    return np.ones(shape=(n_agents, n_agents))


@gin.configurable(module="Luzie")
def custom_adjacency_matrix_builder(adj_matrix: List[List[Union[0,1]]]) -> np.ndarray:
    return np.array(adj_matrix)


@gin.configurable(module="Luzie")
def random_normal_agent_builder(n_agents: int, n_options: int) -> np.ndarray:
    """Initializes the states of all the agents using a Normal distribution."""
    return np.random.normal(size=(n_agents, n_options))


@gin.configurable(module="Luzie")
def homogenous_parameters_builder(adjacency_matrix: np.ndarray, n_options: int, alpha: float, beta: float,
                                  gamma: float, delta: float, d: float, u: float,
                                  v: float, b: float) -> Dict[str, np.ndarray]:
    """
    Sets all the parameters and tensors to update a Luzie-style network.
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


@gin.configurable(module="Luzie")
def agent_types_parameters_builder(adjacency_matrix: np.ndarray, n_options: int,
                                   agent_types: List[AgentType]) -> Dict[str, np.ndarray]:
    """
    Sets all the parameters and tensors to update a General Opinion Dynamics-style network.
    
    The agent_types list that is provided should be of length = n_agents. Each entry denotes the AgentType
    to be used for that specific agent. The AgentTypes are dictionaries that define all the variables and constants
    to be used for that type of agent.
    """
    # Adjacency Tensor
    n_agents = adjacency_matrix.shape[0]
    adjacency_tensor = np.ones(shape=(n_agents, n_agents, n_options, n_options))
    
    for agent in range(n_agents):
        adjacency_tensor[agent] = agent_types[agent]["delta"]
        option_diag = np.diagonal(adjacency_tensor[agent], axis1=1, axis2=2)
        option_diag.setflags(write=1)  # np.diagonal return value is read-only view of input array
        option_diag[:, :] = agent_types[agent]["gamma"]
        adjacency_tensor[agent, agent] = agent_types[agent]["beta"]     # Agent diaognal
        both_diag = np.diagonal(adjacency_tensor[agent, agent], axis1=0, axis2=1)
        both_diag.setflags(write=1)  # np.diagonal return value is read-only view of input array
        both_diag[:] = agent_types[agent]["alpha"]
    
    adjacency_tensor = np.einsum('ijkl,ij->ijkl', adjacency_tensor, adjacency_matrix)

    # Update rule parameters
    resistance = np.ones(shape=(n_agents, n_options)) * [agent["d"] if isinstance(agent["d"], list) else [agent["d"]] for agent in agent_types]
    same_option_attention = np.ones(shape=(n_agents, 1)) * [[agent["u"]] for agent in agent_types]
    other_options_attention = np.ones(shape=(n_agents, 1)) * [[agent["v"]] for agent in agent_types]
    inputs = np.ones(shape=(n_agents, n_options)) * [agent["b"] if isinstance(agent["b"], list) else [agent["b"]] for agent in agent_types]

    params = {
        "adjacency_tensor": adjacency_tensor,
        "d": resistance,
        "u": same_option_attention,
        "v": other_options_attention,
        "b": inputs
    }

    return params

