from absl import app
from absl import flags
from absl import logging

import gin
from itertools import product
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union

from social_dynamics import utility
from social_dynamics.agent_networks.agent_network import AgentNetwork
from social_dynamics.agent_networks.god_agent_network.god_agent_network import GODAgentNetwork
from social_dynamics.agent_networks.luzie_agent_network.luzie_agent_network import LuzieAgentNetwork


def generate_experiment_params_batch(series_dir: Path, batch_size: int) -> Union[None, np.ndarray]:
    """Returns a batch of yet to-be-done experiments.
    
    Note that this does not avoid multiple processes trying to work on the same experiment at the same time.
    It also does not avoid the fact that a process might meet an already done experiment.
    
    E.g. process A gets experiment x as 3rd element in its uncompleted batch and process B gets experiment x
    as 17th elements in its uncompleted batch. Process B will have to  check that indeed the experiment has
    already been done by the time it reached it (most likely).

    Args:
        series_dir (Path): Directory where the results of all the experiments in the series are to be stored.

    Returns:
        Union[None, np.ndarray]: None if there are no experiments left to run, list of experiment params
                    otherwise
    """
    experiment_params_array = np.array(list(product(np.linspace(-2, 2, 11), repeat=4)))
    experiment_path_list = [
        series_dir.joinpath(generate_experiment_name(alpha=alpha, beta=beta, gamma=gamma, delta=delta))
        for alpha, beta, gamma, delta in experiment_params_array
    ]

    completed_experiments = np.array(list(series_dir.iterdir()))

    to_do_experiments = experiment_params_array[~np.isin(experiment_path_list, completed_experiments)].tolist(
    )

    # The list gets shuffled to decrease the likelihood that multiple processes spawned at the same time
    # might collide on the same locks.
    rng = np.random.default_rng()
    rng.shuffle(to_do_experiments)

    return to_do_experiments[:batch_size]


def generate_experiment_name(alpha: float, beta: float, gamma: float, delta: float) -> str:
    return "{}alpha_{}beta_{}gamma_{}delta".format(np.round(alpha, 1), np.round(beta, 1), np.round(gamma, 1),
                                                   np.round(delta, 1))


@gin.configurable
def run_experiment(experiment_dir: Path, num_time_steps: int, agent_network: AgentNetwork,
                   checkpoint_interval: int, metrics_interval: int) -> None:
    """Runs a single experiment in the series. Saves the results of the metrics 

    Args:
        experiment_dir (Path): Directory to store the experiment's results.
        num_time_steps (int): Number of time steps to simulate. The time interval between each of these
                time steps is controlled by the AgentNetwork object (in the cases in which the 
                model assumes the existence of a time interval)
        checkpoint_interval (int): The number of time steps after which the data stored in the metrics is saved
                a shorter interval decreases speed of the code while reducing memory footprint
        metrics_interval (int): The interval for metrics' data collection. E.g. one might not be interested to
                know the state of the network at every time step, but at every x time_steps. A longer interval
                increases speed and reduces memory footprint (at the price of less data collected)
    """
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
        files = sorted(metric_dir.iterdir(), key=os.path.getmtime)
        files = [file for file in files]
        results = [np.load(file) for file in files]
        np.save(files[-1], np.concatenate(results, axis=0))
        for file in files[:-1]:
            file.unlink()


def run_experiment_series(root_dir: Path,
                          series_name: str,
                          batch_size: int,
                          random_seed: Optional[int] = None,
                          random_state_path: Optional[str] = None) -> None:
    """Runs a series of experiments iterating through certain parameters as determined by the customizable
    for loops below.

    Args:
        root_dir (Path): The directory where results for all experiments are saved 
                (relative path based on where the terminal is located)
        series_name (str): Name with which to refer to this series and to name its directory of results.
        batch_size (int): Batch size to be use by the loop over experiments.
        random_seed (Optional[int], optional): Random seed to be used for replicability. Defaults to None.
        random_state_path (Optional[str], optional): Path to the random state to be loaded and used as seed for replicability.
                Defaults to None.
    """
    assert (random_seed is None) or (random_state_path is None), (
        "Should not feed a random seed and a random state at the same time."
        " Only one of the two can be used at once")
    root_dir = root_dir.expanduser()
    series_dir = root_dir.joinpath(series_name)
    if not series_dir.is_dir():
        series_dir.mkdir(parents=True)

    # Managing the random state for replicabiity purposes
    #FIXME the random state isn't properly managed when running multiple terminals
    # on the same experiment series for parallel execution. It is overwritten by the last
    # terminal that is instantiated
    if random_state_path is not None:
        random_state = tuple(np.load(random_state_path, allow_pickle=True))
        np.random.set_state(random_state)
    else:
        np.random.seed(random_seed)
    random_state = np.random.get_state()
    np.save(series_dir.joinpath('initial_random_state.npy'), np.array(random_state, dtype='object'))

    # The reason for sampling batches multiple times instead of just iterating through everything is that
    # this way we regularly clearup any experiments already done. This saves on a lot of I/O operations
    # to the disk, particularly towards the end when there can be thousands of folders already done.
    experiment_params_batch = generate_experiment_params_batch(series_dir=series_dir, batch_size=batch_size)
    while experiment_params_batch:
        for alpha, beta, gamma, delta in tqdm(experiment_params_batch):
            agent_network = LuzieAgentNetwork(
                builders_kwargs={
                    "adj_matrix_builder_kwargs": dict(),
                    "agents_builder_kwargs": dict(),
                    "parameters_builder_kwargs": {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "delta": delta
                    }
                })
            experiment_name = generate_experiment_name(alpha=alpha, beta=beta, gamma=gamma, delta=delta)

            experiment_dir = series_dir.joinpath(experiment_name)

            if not utility.check_lock(experiment_dir):
                continue

            utility.acquire_lock(experiment_dir)

            run_experiment(experiment_dir=experiment_dir,
                           agent_network=agent_network,
                           metrics_interval=50)

            utility.release_lock(experiment_dir)

        experiment_params_batch = generate_experiment_params_batch(series_dir=series_dir, batch_size=batch_size)


def main(_) -> None:
    logging.set_verbosity(logging.INFO)
    utility.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)

    run_experiment_series(root_dir=Path(FLAGS.root_dir),
                          series_name=FLAGS.series_name,
                          batch_size=FLAGS.batch_size)


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
    flags.DEFINE_integer('batch_size', 10, 'Batch size for the experiment loop.')
    flags
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('series_name')
    flags.mark_flag_as_required('gin_files')

    app.run(main)
