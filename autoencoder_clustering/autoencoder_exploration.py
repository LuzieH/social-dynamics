from absl import app
from absl import flags
from absl import logging

from functools import partial
import numpy as np
import os
from pathlib import Path
import pickle

from social_dynamics import utility
from social_dynamics import autoencoder_utils

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import MSE


LOGGING_DICT = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}
downsampling = 4


def generate_experiment_name(key: str, model_index: int) -> str:
    return "{}-{}".format(key, model_index)


def run_autoencoder_exploration(root_dir: str, series_dir: str, batch_size: int) -> None:
    root_dir_path = Path(root_dir)
    results_dir_path = root_dir_path.joinpath("autoencoders_results")
    series_dir_path = Path(series_dir)

    datasets, n_agents, n_options = autoencoder_utils.load_all_datasets(series_dir=series_dir_path,
                                                                        downsampling=downsampling)

    input_shapes = autoencoder_utils.determine_input_shapes(datasets)

    with open(root_dir_path.joinpath('models_kwargs.pickle'), 'rb') as handle:
        models_kwargs = pickle.load(handle)

    keys = list(models_kwargs.keys())
    rng = np.random.default_rng()
    rng.shuffle(keys)
    for key in keys:
        logging.info("Starting to work on key: {}".format(key))
        model_type, input_type = key.split("-")
        experiment_params_list = [{"key": key,"model_index": model_index}
                                  for model_index in range(len(models_kwargs[key]))]

        dataset = datasets[key].shuffle(buffer_size=20_000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
        y_true = np.array(list(datasets[key].as_numpy_iterator()))[:, 1]
        input_shape = input_shapes[key]

        batch_generator = partial(utility.generate_experiment_params_batch,
                                  all_results_dir=results_dir_path,
                                  experiment_params_list=experiment_params_list,
                                  experiment_name_generator=generate_experiment_name,
                                  batch_size=batch_size)
        experiment_params_batch = batch_generator()
        while experiment_params_batch:
            for experiment_params in experiment_params_batch:
                model_index = experiment_params["model_index"]
                experiment_name = generate_experiment_name(key, model_index)

                model_results_path = results_dir_path.joinpath(experiment_name)
                if not utility.check_lock(model_results_path):
                    continue

                utility.acquire_lock(model_results_path)

                logging.info("Acquired lock on {}".format(experiment_name))

                model_kwargs = models_kwargs[key][model_index]
                if model_type == "cnn":
                    model = autoencoder_utils.get_cnn_autoencoder_model(input_shape,
                                                                        **model_kwargs,
                                                                        sigmoid=False)
                else:
                    model = autoencoder_utils.get_dnn_autoencoder_model(input_shape,
                                                                        **model_kwargs,
                                                                        sigmoid=False)

                model_hist = model.fit(dataset, epochs=30, verbose=0)
                y_pred = model.predict(y_true)
                # CNNs have bidimensional output (time x agent-option_channel) and thus MSE needs
                # to be averaged over two dimensions and not just one.
                if (model_type == "cnn") or ("batched" in input_type):
                    mses = np.mean(MSE(y_true, y_pred).numpy(), axis=-1)
                else:
                    mses = MSE(y_true, y_pred).numpy()
                
                encoder = Model(model.input, model.get_layer('embedding').output)
                embeddings = encoder.predict(y_true)

                model_results_path.mkdir(parents=True, exist_ok=True)
                np.save(model_results_path.joinpath("history.npy"), model_hist.history)
                np.save(model_results_path.joinpath("mses.npy"), mses)
                np.save(model_results_path.joinpath("predictions.npy"), y_pred)
                np.save(model_results_path.joinpath("embeddings.npy"), embeddings)
                # Batched dnn outputs have shape (n_samples, n_agents*n_options, n_timesteps)
                # The last two dimensions must be swapped otherwise plotting is going to be off.
                trues = y_true if "batched" not in input_type else np.transpose(y_true, axes=(0, 2, 1))
                preds = y_pred if "batched" not in input_type else np.transpose(y_pred, axes=(0, 2, 1))
                autoencoder_utils.generate_prediction_plots(y_true=trues,
                                                            y_pred=preds,
                                                            mses=mses,
                                                            n_agents=n_agents,
                                                            n_options=n_options,
                                                            save_path=model_results_path)

                tf.keras.backend.clear_session()

                utility.release_lock(model_results_path)
                logging.info("Saved and released lock on {}".format(experiment_name))

            experiment_params_batch = batch_generator()


def main(_) -> None:
    logging.set_verbosity(LOGGING_DICT[FLAGS.logging])

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    run_autoencoder_exploration(root_dir=FLAGS.root_dir,
                                series_dir=FLAGS.series_dir,
                                batch_size=FLAGS.batch_size)


if __name__ == '__main__':
    flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing results.')
    flags.DEFINE_string('series_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Path to the experiment series to train the autoencoders on.')
    flags.DEFINE_integer('batch_size', 10, 'Batch size for the experiment loop.')
    flags.DEFINE_string('logging', 'warning', 'Logging level. Must be a string in '
                        '["debug", "info", "warning"]')
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('series_dir')

    app.run(main)
