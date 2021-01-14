from typing import List
import gin
from . import agent_network



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
def setup_network() -> agent_network.AgentNetwork:
    raise NotImplementedError()


@gin.configurable
def compute_metrics(agent_network):
    raise NotImplementedError()


def save_metrics(metrics, experiment_dir):
    raise NotImplementedError()

