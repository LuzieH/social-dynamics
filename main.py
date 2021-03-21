from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import os
from tqdm import tqdm
from typing import List, Optional

from social_dynamics import utility
from social_dynamics.metrics import metric


@gin.configurable
def run_experiment(root_dir: str,
                   experiment_name: str,
                   num_time_steps: int,
                   metrics: List[metric.Metric],
                   checkpoint_interval: int,
                   metrics_interval: int,
                   random_seed: Optional[int] = None,
                   random_state_path: Optional[str] = None) -> None:
    """Runs the experiment

    Args:
        root_dir (str): The directory where results for all experiments are saved 
                (relative path based on where the terminal is located)
        experiment_name (str): unique identifier for the experiment configuration which is used to make
                a results folder for this experiment run
        num_time_steps (int): Number of time steps to simulate. The time interval between each of these
                time steps is controlled by the AgentNetwork object (in the cases in which the 
                model assumes the existence of a time interval)
        metrics (List[metric.Metric]): List of metric.Metric instances to be keep track of any relevant data during simulation
        checkpoint_interval (int): The number of time steps after which the data stored in the metrics is saved
                a shorter interval decreases speed of the code while reducing memory footprint
        metrics_interval (int): The interval for metrics' data collection. E.g. one might not be interested to
                know the state of the network at every time step, but at every x time_steps. A longer interval
                increases speed and reduces memory footprint (at the price of less data collected)
        random_seed (Optional[int], optional): Random seed to be used for replicability. Defaults to None.
        random_state_path (Optional[str], optional): Path to the random state to be loaded and used as seed for replicability.
                Defaults to None.
    """
    assert (random_seed is None) or (random_state_path is None), (
        "Should not feed a random seed and a random state at the same time."
        " Only one of the two can be used at once")
    root_dir = os.path.expanduser(root_dir)
    experiment_dir = os.path.join(root_dir, experiment_name)

    if not os.path.isdir(experiment_dir):
        experiment_run_dir = os.path.join(experiment_dir, '0')
        os.makedirs(experiment_run_dir)
    else:
        run_id = str(max([int(folder) for folder in os.listdir(experiment_dir)]) + 1)
        experiment_run_dir = os.path.join(experiment_dir, run_id)
        os.makedirs(experiment_run_dir)

    # Managing the random state for replicabiity purposes
    if random_state_path is not None:
        random_state = tuple(np.load(random_state_path, allow_pickle=True))
        np.random.set_state(random_state)
    else:
        np.random.seed(random_seed)
    random_state = np.random.get_state()
    np.save(os.path.join(experiment_run_dir, 'initial_random_state.npy'),
            np.array(random_state, dtype='object'))

    agent_network = utility.setup_network()

    for t in tqdm(range(1, num_time_steps + 1)):

        agent_network.step()

        if t % metrics_interval == 0:
            for metric in metrics:
                metric(agent_network)

        if t % checkpoint_interval == 0:
            for metric in metrics:
                metric.save(save_path=experiment_run_dir, time_step=t)
                metric.reset()


def main(_) -> None:
    logging.set_verbosity(logging.INFO)
    utility.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    run_experiment(root_dir=FLAGS.root_dir)


if __name__ == '__main__':
    flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing results of the metrics.')
    flags.DEFINE_multi_string(
        'gin_files', [], 'List of paths to gin configuration files (e.g.'
        '"configs/homogenous_luzie_net.gin").')
    flags.DEFINE_multi_string(
        'gin_bindings', [], 'Gin bindings to override the values set in the config files '
        '(e.g. "run_experiment.random_seed=42").')
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('root_dir')

    app.run(main)
