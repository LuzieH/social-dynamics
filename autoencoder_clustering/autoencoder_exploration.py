from absl import app
from absl import flags
from absl import logging

import numpy as np
import os
from pathlib import Path
import pickle

from social_dynamics import utility
from social_dynamics.autoencoder_utils import determine_input_shapes, load_all_datasets
from social_dynamics.autoencoder_utils import get_dnn_autoencoder_model, get_cnn_autoencoder_model

import tensorflow as tf
from tensorflow.keras.losses import MSE

from tqdm import tqdm


downsampling = 4


def generate_experiment_name(key: str, model_index: int) -> str:
    return "{}-{}".format(key, model_index)


def main(_) -> None:
    logging.set_verbosity(logging.INFO)
    root_dir = Path(FLAGS.root_dir)
    series_dir = Path(FLAGS.series_dir)
    
    
    datasets = load_all_datasets(series_dir=series_dir)
    
    input_shapes = determine_input_shapes(datasets)
    
    with open('models_kwargs.pickle', 'rb') as handle:
        models_kwargs = pickle.load(handle)
    
    
    keys = models_kwargs.keys()
    rng = np.random.default_rng()
    rng.shuffle(keys)
    for key in keys:
        model_type, input_type = key.split("-")
        
        
        dataset = datasets[key].shuffle(buffer_size=20_000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
        y_true = np.array(list(datasets[key].as_numpy_iterator()))[:, 1]
        input_shape = input_shapes[key]
        
        experiment_params_list = [(key, model_index) for model_index in range(len(models_kwargs[key]))]
        experiment_params_batch = utility.generate_experiment_params_batch(
            all_results_dir=root_dir,
            experiment_params_list=experiment_params_list,
            experiment_name_generator=generate_experiment_name,
            batch_size=FLAGS.batch_size)
        while experiment_params_batch:
            for _, model_index in tqdm(experiment_params_batch):
                model_kwargs = models_kwargs[key][model_index]
                if model_type == "cnn":
                    model = get_cnn_autoencoder_model(input_shape, **model_kwargs, sigmoid=False)
                else:
                    model = get_dnn_autoencoder_model(input_shape, **model_kwargs, sigmoid=False)
                
                
                model_hist = model.fit(dataset, epochs=30, verbose=0)
                
                experiment_name = generate_experiment_name(key, model_index)
                model_results_path = root_dir.joinpath(experiment_name)
                model_results_path.mkdir(parents=True, exist_ok=True)
                np.save(model_results_path.joinpath("history.npy"), model_hist.history)
                
                
                y_preds = model.predict(y_true)
                mses = MSE(y_true, y_preds).numpy()
                np.save(model_results_path.joinpath("mses.npy"), mses)
                
                tf.keras.backend.clear_session()
            
            experiment_params_batch = utility.generate_experiment_params_batch(
                all_results_dir=root_dir,
                experiment_params_list=experiment_params_list,
                experiment_name_generator=generate_experiment_name,
                batch_size=FLAGS.batch_size)
    


if __name__ == '__main__':
    flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing results.')
    flags.DEFINE_string('series_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Path to the experiment series to train the autoencoders on.')
    flags.DEFINE_integer('batch_size', 10, 'Batch size for the experiment loop.')
    flags
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('series_dir')
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    app.run(main)

