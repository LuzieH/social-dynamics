from typing import Callable
import numpy as np
import time

from numpy.lib.twodim_base import diag

def activation_function_builder(a: float, b: float, c: float):
    
    def activation_function(x: np.ndarray) -> np.ndarray:
        return a*np.tanh(b*x + c*np.tanh(x**2))
    
    return activation_function


def compute_correct_result(adjacency_matrix: np.ndarray, agents: np.ndarray, attention: np.ndarray,
                           S1: Callable[[np.ndarray], np.ndarray], S2: Callable[[np.ndarray], np.ndarray]):
    result = []
    n_agents, n_options = agents.shape
    for i in range(n_agents):
        agent_attentions = []
        for j in range(n_options):
            S1_term = 0
            for k in range(n_agents):
                S1_term += adjacency_matrix[i, k , j, j] * agents[k, j]
            S1_term = S1(S1_term)
            
            S2_term = 0
            for l in range(n_options):
                if l != j:
                    sub_S2_term = 0
                    for k in range(n_agents):
                        sub_S2_term += adjacency_matrix[i, k, j, l] * agents[k, l]
                    S2_term += S2(sub_S2_term)
            
            agent_attentions.append(attention[i][0] * (S1_term + S2_term))
        
        result.append(agent_attentions)
    
    return np.array(result)


def my_code(adjacency_matrix: np.ndarray, agents: np.ndarray, attention: np.ndarray,
            S1: Callable[[np.ndarray], np.ndarray], S2: Callable[[np.ndarray], np.ndarray]):
    
    return attention*(S1(np.sum(np.diagonal(adjacency_matrix, axis1=2, axis2=3) * agents,
                                axis=1))
                      +
                      np.sum(S2(np.sum(adjacency_matrix * agents[:, np.newaxis, :], axis=1)),
                             axis=2, where=non_diag_bool_tensor))


def luzie_code(adjacency_matrix: np.ndarray, agents: np.ndarray, attention: np.ndarray,
               S1: Callable[[np.ndarray], np.ndarray], S2: Callable[[np.ndarray], np.ndarray]):
    
    return attention*(S1(np.einsum('ikj,kj->ij',np.einsum('...ii->...i',adjacency_matrix), agents)) +
                      np.sum(S2(np.einsum('ikjl,kl->ijl', adjacency_matrix, agents)), axis=2) -
                      np.diagonal(S2(np.einsum('ikjl,kl->ijl', adjacency_matrix, agents)),axis1=1, axis2=2)
                      )


def modified_luzie_code(adjacency_matrix: np.ndarray, agents: np.ndarray, attention: np.ndarray,
                        S1: Callable[[np.ndarray], np.ndarray], S2: Callable[[np.ndarray], np.ndarray]):
    
    return attention*(S1(np.einsum('ikj,kj->ij',np.einsum('...ii->...i',adjacency_matrix), agents)) +
                      np.sum(S2(np.einsum('ikjl,kl->ijl', adjacency_matrix, agents)),
                             axis=2, where=non_diag_bool_tensor))


def diag_mod_luzie_code(adjacency_matrix: np.ndarray, agents: np.ndarray, attention: np.ndarray,
                        S1: Callable[[np.ndarray], np.ndarray], S2: Callable[[np.ndarray], np.ndarray]):
    
    return attention*(S1(np.einsum('ikj,kj->ij', np.diagonal(adjacency_matrix, axis1=2, axis2=3), agents)) +
                      np.sum(S2(np.einsum('ikjl,kl->ijl', adjacency_matrix, agents)),
                             axis=2, where=non_diag_bool_tensor))


N_AGENTS = 500
N_OPTIONS = 10
n = 1
N = 10000


S1 = activation_function_builder(1, 1, 0)
S2 = activation_function_builder(0.5, 2, 0)
non_diag_bool_tensor = np.ones(shape=(N_AGENTS, N_OPTIONS, N_OPTIONS), dtype=np.bool)
for option in range(N_OPTIONS): 
    non_diag_bool_tensor[:, option, option] = False


for z in range(n):
    print(z)
    attention = np.random.uniform(size=(N_AGENTS, 1)) 
    agents = np.random.normal(size=(N_AGENTS, N_OPTIONS))
    adjacency_matrix = np.random.normal(size=(N_AGENTS, N_AGENTS, N_OPTIONS, N_OPTIONS))

    true_result = compute_correct_result(adjacency_matrix, agents, attention, S1, S2)
    my_result = my_code(adjacency_matrix, agents, attention, S1, S2)
    luzie_result = luzie_code(adjacency_matrix, agents, attention, S1, S2)
    modified_luzie_result = modified_luzie_code(adjacency_matrix, agents, attention, S1, S2)

    assert (np.isclose(true_result, my_result).all())
    assert (np.isclose(true_result, luzie_result).all())
    assert (np.isclose(true_result, modified_luzie_result).all())


attention = np.random.uniform(size=(N_AGENTS, 1)) 
agents = np.random.normal(size=(N_AGENTS, N_OPTIONS))
adjacency_matrix = np.random.normal(size=(N_AGENTS, N_AGENTS, N_OPTIONS, N_OPTIONS))


t0 = time.time()
for i in range(N):
    result = my_code(adjacency_matrix, agents, attention, S1, S2)

print('My code:', time.time() - t0)


t0 = time.time()
for i in range(N):
    result = luzie_code(adjacency_matrix, agents, attention, S1, S2)

print('Luzie code:', time.time() - t0)


t0 = time.time()
for i in range(N):
    result = modified_luzie_code(adjacency_matrix, agents, attention, S1, S2)

print('Modified Luzie code:', time.time() - t0)


t0 = time.time()
for i in range(N):
    result = diag_mod_luzie_code(adjacency_matrix, agents, attention, S1, S2)

print('Modified Luzie code with np.diagonal:', time.time() - t0)



t0 = time.time()
for i in range(N):
    result = compute_correct_result(adjacency_matrix, agents, attention, S1, S2)

print('For-loop code:', time.time() - t0)
