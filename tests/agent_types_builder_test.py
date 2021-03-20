from re import X
import numpy as np
from social_dynamics.agent_networks.luzie_agent_network import builders

n_agents = 4
n_options = 3

agent_type_1 = {
    "alpha": 0.1,       # 
    "beta": 0.2,        #
    "gamma": -0.3,      #
    "delta": 0.4,       #
    "d": 3,             #
    "u": 0.5,           #
    "v": 0.6,           #
    "b": 7              #
}

agent_type_2 = {
    "alpha": 0.8,          # 
    "beta": 0.9,           #
    "gamma": -0.11,         #
    "delta": 0.12,          #
    "d": 4,                #
    "u": 0.13,              #
    "v": 0.14,              #
    "b": 0                 #
}


agent_types = [agent_type_1, agent_type_2, agent_type_1, agent_type_2]

adjacency_matrix = builders.complete_adjacency_matrix_builder(n_agents=n_agents)

params = builders.agent_types_parameters_builder(adjacency_matrix, n_options, agent_types)

adjacency_tensor = params["adjacency_tensor"]


for agent1 in range(n_agents):
    for agent2 in range (n_agents):
        for option1 in range(n_options):
            for option2 in range(n_options):
                if (agent1 == agent2) and (option1 == option2):
                    assert adjacency_tensor[agent1, agent2, option1, option2] == agent_types[agent1]["alpha"]
                elif agent1 == agent2:
                    assert adjacency_tensor[agent1, agent2, option1, option2] == agent_types[agent1]["beta"]
                elif option1 == option2:
                    assert adjacency_tensor[agent1, agent2, option1, option2] == agent_types[agent1]["gamma"]
                else:
                    assert adjacency_tensor[agent1, agent2, option1, option2] == agent_types[agent1]["delta"]


