from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from social_dynamics import utility
from social_dynamics.agent_networks.agent_network import AgentNetwork
from social_dynamics.agent_networks.god_agent_network.god_agent_network import GODAgentNetwork
from social_dynamics.agent_networks.luzie_agent_network.luzie_agent_network import LuzieAgentNetwork


@gin.configurable
def run_experiment(series_dir: Path,
                   experiment_name: str,
                   num_time_steps: int,
                   agent_network: AgentNetwork,
                   checkpoint_interval: int,
                   metrics_interval: int) -> None:
    """Runs a single experiment in the series. Saves the results of the metrics 

    Args:
        series_dir (str): Identifier of the series of experiments that is being run.
        experiment_name (str): unique identifier for the experiment configuration which is used to make
                a results folder for this experiment run
        num_time_steps (int): Number of time steps to simulate. The time interval between each of these
                time steps is controlled by the AgentNetwork object (in the cases in which the 
                model assumes the existence of a time interval)
        checkpoint_interval (int): The number of time steps after which the data stored in the metrics is saved
                a shorter interval decreases speed of the code while reducing memory footprint
        metrics_interval (int): The interval for metrics' data collection. E.g. one might not be interested to
                know the state of the network at every time step, but at every x time_steps. A longer interval
                increases speed and reduces memory footprint (at the price of less data collected)
    """
    experiment_dir = series_dir.joinpath(experiment_name)
    if not experiment_dir.is_dir():
        experiment_dir.mkdir(parents=True)
    
    metrics = utility.setup_metrics(checkpoint_interval=checkpoint_interval,
                                    metrics_interval=metrics_interval)
    
    for t in range(1, num_time_steps + 1):

        agent_network.step()

        if t % metrics_interval == 0:
            for metric in metrics:
                metric(agent_network)

        if t % checkpoint_interval == 0:
            for metric in metrics:
                metric.save(save_path=experiment_dir, time_step=t)
                metric.reset()
    
    for metric_dir in experiment_dir.iterdir():
        dir_path = experiment_dir.joinpath(metric_dir)
        files = sorted(dir_path.iterdir(), key=os.path.getmtime)
        files = [file for file in files]
        results = [np.load(file) for file in files]
        np.save(files[-1], np.concatenate(results, axis=0))
        for file in files[:-1]:
            file.unlink()


def run_experiment_series(root_dir: Path,
                          series_name: str,
                          random_seed: Optional[int] = None,
                          random_state_path: Optional[str] = None) -> None:
    """Runs a series of experiments iterating through certain parameters as determined by the customizable
    for loops below.

    Args:
        root_dir (str): The directory where results for all experiments are saved 
                (relative path based on where the terminal is located)
        random_seed (Optional[int], optional): Random seed to be used for replicability. Defaults to None.
        random_state_path (Optional[str], optional): Path to the random state to be loaded and used as seed for replicability.
                Defaults to None.
    """
    assert (random_seed is None) or (random_state_path is None), (
        "Should not feed a random seed and a random state at the same time."
        " Only one of the two can be used at once")
    root_dir = root_dir.expanduser()
    series_dir = root_dir.joinpath(series_name)
    if not series_dir.isdir():
        series_dir.mkdir(parents=True)

    # Managing the random state for replicabiity purposes
    if random_state_path is not None:
        random_state = tuple(np.load(random_state_path, allow_pickle=True))
        np.random.set_state(random_state)
    else:
        np.random.seed(random_seed)
    random_state = np.random.get_state()
    np.save(series_dir.joinpath('initial_random_state.npy'),
            np.array(random_state, dtype='object'))
    
    for alpha in np.linspace(-2, 2, 11):
        for beta in tqdm(np.linspace(-2, 2, 11)):
            for gamma in np.linspace(-2, 2, 11):
                for delta in np.linspace(-2, 2, 11):
                    agent_network = LuzieAgentNetwork(builders_kwargs={"adj_matrix_builder_kwargs": dict(),
                                                                       "agents_builder_kwargs": dict(),
                                                                       "parameters_builder_kwargs": {"alpha": alpha,
                                                                                                     "beta": beta,
                                                                                                     "gamma": gamma,
                                                                                                     "delta": delta}
                                                                    })
                    experiment_name = "{}alpha_{}beta_{}gamma_{}delta".format(np.round(alpha, 1), np.round(beta, 1),
                                                                              np.round(gamma, 1), np.round(delta, 1))
                    if not series_dir.joinpath(experiment_name).exists():
                        run_experiment(series_dir=series_dir,
                                       experiment_name=experiment_name,
                                       agent_network=agent_network,
                                       metrics_interval=50)



def main(_) -> None:
    logging.set_verbosity(logging.INFO)
    utility.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    
    run_experiment_series(root_dir=Path(FLAGS.root_dir), series_name=FLAGS.series_name)


if __name__ == '__main__':
    flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing results of the metrics.')
    flags.DEFINE_string('series_name', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Name to identify the experiment series')
    flags.DEFINE_multi_string(
        'gin_files', [], 'List of paths to gin configuration files (e.g.'
        '"configs/homogenous_luzie_net.gin").')
    flags.DEFINE_multi_string(
        'gin_bindings', [], 'Gin bindings to override the values set in the config files '
        '(e.g. "run_experiment.random_seed=42").')
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('root_dir')

    app.run(main)
