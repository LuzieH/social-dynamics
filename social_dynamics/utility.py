import gin
import numpy as np
from pathlib import Path
from social_dynamics import agent_networks
from social_dynamics.agent_networks import god_agent_network, luzie_agent_network
from social_dynamics.metrics.metric import Metric
from typing import Dict, List


IMPLEMENTED_MODELS = ['general_opinion_dynamics', 'luzie_network']
LOCKS_PATH = Path("locks")


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
        raise ValueError("Expected gin_files paths to be like {}, instead got {}".format(
            '.../configs/...', gin_files[0]))

    configs_folder_path = '/'.join(path_folders[:configs_folder_index + 1])
    gin.add_config_file_search_path(configs_folder_path)
    gin.parse_config_files_and_bindings(gin_files, bindings=gin_bindings, skip_unknown=False)


@gin.configurable
def setup_network(model: str) -> agent_networks.agent_network.AgentNetwork:
    if model == 'general_opinion_dynamics':
        return god_agent_network.god_agent_network.GODAgentNetwork()
    elif model == 'luzie_network':
        return luzie_agent_network.luzie_agent_network.LuzieAgentNetwork()
    raise ValueError('Expected argument "model" to be a string in {}'.format(IMPLEMENTED_MODELS))


@gin.configurable
def setup_metrics(checkpoint_interval: int, metrics_interval: int, metrics: List[type]) -> List[Metric]:
    """Instantiates all the required metrics for the run.
        
    Args:
        checkpoint_interval (int): Interval between checkpoints when running an experiment. It is used 
                together with the metrics interval parameter to determine an appropriate buffer_size 
                for the metrics.
        metrics_interval (int): Interval between calls to the metrics to evaluate the AgentNetwork and
                store relevant results.
        metrics (List[type]): List of metric classes that will be instantiated and returned.
                This parameter is meant to be passed via Gin Config.

    Raises:
        ValueError: If checkpoint_interval is not a multiple of the metrics_interval

    Returns:
        List[metric.Metric]: List of metrics that will be used during the experiment
    """
    if checkpoint_interval % metrics_interval != 0:
        raise ValueError("Checkpointing interval is supposed to be a multiple of the metrics interval. "
                         f"Got instead checkpoint_interval:{checkpoint_interval} "
                         f"and metrics_interval: {metrics_interval}")
    
    buffer_size = checkpoint_interval//metrics_interval
    
    built_metrics = [metric(buffer_size=buffer_size) for metric in metrics]
    
    return built_metrics


def check_lock(results_path: Path) -> bool:
    """
    Checks if the run defined by the results_path passed needs to be executed, and if so
    checks that there isn't a lock already on it.

    Returns:
        bool: Whether to execute the run or not.
    """
    lock_name = results_path.name + ".npy"
    lock_path = LOCKS_PATH.joinpath(lock_name)
    if results_path.exists() or lock_path.exists():
        return False

    return True


def acquire_lock(results_path: Path) -> None:
    """
    Adds a lock on the current run.
    """
    if not LOCKS_PATH.exists():
        LOCKS_PATH.mkdir()
    lock_name = results_path.name + ".npy"
    lock_path = LOCKS_PATH.joinpath(lock_name)
    np.save(lock_path, None)


def release_lock(results_path: Path) -> None:
    """
    Releases the lock on the current run.
    """
    lock_name = results_path.name + ".npy"
    lock_path = LOCKS_PATH.joinpath(lock_name)
    lock_path.unlink()


