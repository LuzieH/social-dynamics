from typing import List
import gin
from social_dynamics import agent_networks


IMPLEMENTED_MODELS = ['general_opinion_dynamics']

def load_gin_configs(gin_files: List[str], gin_bindings: List[str]) -> None:
    """Loads gin configuration files.

    Args:
      gin_files: A list of paths to the gin configuration files for this
        experiment.
      gin_bindings: List of gin parameter bindings to override the values in the
        config files.
    """
    try:
        path_folders = gin_files[0].split('/')
        configs_folder_index = path_folders.index('configs')
    except:
        raise ValueError("Expected gin_files paths to be like {}, instead got {}".format('.../configs/...', gin_files[0]))
    
    configs_folder_path = '/'.join(path_folders[:configs_folder_index + 1])
    gin.add_config_file_search_path(configs_folder_path)
    gin.parse_config_files_and_bindings(gin_files, bindings=gin_bindings, skip_unknown=False)


@gin.configurable
def setup_network(model: str) -> agent_networks.agent_network.AgentNetwork:
    if model == 'general_opinion_dynamics':
        return agent_networks.god_agent_network.GODAgentNetwork()
    raise ValueError('Expected argument "model" to be a string in {}'.format(IMPLEMENTED_MODELS))


@gin.configurable
def compute_metrics(agent_network: agent_networks.agent_network.AgentNetwork):
    raise NotImplementedError()


def save_metrics(metrics, experiment_dir):
    raise NotImplementedError()

