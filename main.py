import os
import time
from typing import List
import numpy as np

from absl import app
from absl import flags
from absl import logging

import gin

from social_dynamics import utility
from social_dynamics.metrics import metric



flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_files', [], 'List of paths to gin configuration files (e.g.'
                          '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [], 'Gin bindings to override the values set in the config files '
    '(e.g. "train_eval.num_iterations=100").')

FLAGS = flags.FLAGS



@gin.configurable
def run_experiment(
        root_dir: str,
        experiment_name: str,
        num_time_steps: int,
        metrics: List[metric.Metric],
        checkpoint_interval: int,
        metrics_interval: int,
        # random_seed: int = None
) -> None:
    """
    Runs the experiment
    
    Args:
        root_dir - the directory where results for all experiments are saved 
                (relative path based on where the terminal is located)
        experiment_name - unique identifier for the experiment id which is used to make
                a results folder for this experiment run
    
    """
    root_dir = os.path.expanduser(root_dir)
    experiment_dir = os.path.join(root_dir, experiment_name)

    # TODO create code that supports running the same experiment multiple times and saving all results
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    
    for metric in metrics:
        os.mkdir(os.path.join(experiment_dir, metric.name))
    # TODO use the random seed only if given. Make sure that it effects every source of randomness
    
    agent_network = utility.setup_network()

    for t in range(1, num_time_steps + 1):
        
        agent_network.step()

        if t % metrics_interval == 0:
            for metric in metrics:
                metric(agent_network)
        
        if t % checkpoint_interval == 0:
            for metric in metrics:
                np.save(os.path.join(experiment_dir, metric.name, 'results_t{}.npy'.format(t)), metric.result())


def main(_) -> None:
    logging.set_verbosity(logging.INFO)
    utility.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    run_experiment(root_dir=FLAGS.root_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
