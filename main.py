import os
import time

from absl import app
from absl import flags
from absl import logging

import gin

from . import utility



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
        checkpoint_interval: int,
        metrics_interval: int,
        random_seed: int = None
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

    # TODO create results directory
    # TODO use the random seed only if given. Make sure that it effects every source of randomness
    
    agent_network = utility.setup_network()

    for t in range(num_time_steps):
        
        agent_network.step()
        
        # Checkpointing and flushing summaries
        if t % checkpoint_interval == 0:
            # checkpoint current state. Do we even need to do this?
            pass

        # Evaluation Run
        if t % metrics_interval == 0:
            # Compute the metrics
            metrics = utility.compute_metrics(agent_network)
            utility.save_metrics(metrics, experiment_dir)

def main(_) -> None:
    logging.set_verbosity(logging.INFO)
    utility.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    run_experiment(root_dir=FLAGS.root_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
