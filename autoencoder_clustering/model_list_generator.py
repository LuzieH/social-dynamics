from absl import app
from absl import flags
from absl import logging

from itertools import product
import numpy as np
from pathlib import Path
import pickle

from social_dynamics.autoencoder_utils import create_dataset
from social_dynamics.autoencoder_utils import compute_embedding_length
from social_dynamics.autoencoder_utils import get_dnn_autoencoder_model, get_cnn_autoencoder_model

import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple

downsampling = 4
ModelParams = List[Dict[str, float]]


def generate_model_kwargs_lists() -> List[Tuple[str, int, ModelParams]]:
    """Generate the list of autoencoder models to be trained.

    Returns:
        List[Tuple[str, int, ModelParams]]: List contiaining the paramters for all the autoencoder models
                to be trained and the required identifiers to iterate through them.
    """
    # Generate the list of CNN-type autoencoders.
    # This is more complex than for the DNNs because the space of hyperparameters is much much larger
    # and not all possible choices are allowed. What is allowed is determined via compute_embedding_length()
    cnn_kwargs_list = []
    rng = np.random.default_rng(42)
    for n_layers in range(6, 10):
        for stridess in tqdm(list(product((1, 2), repeat=n_layers))):
            # This n_samples value has been empirically determined to make ti so that we get ~200 models
            # for every n_layer. This makes the total number of CNNs comparable to the DNNs.
            n_samples = int(512/2.2**(n_layers-6))
            for kernel_sizes in rng.choice(np.arange(3, 10), (n_samples, n_layers)):
                layers_kwargs = [{"kernel_size": kernel_size,
                                  "strides": strides}
                                 for kernel_size, strides in zip(kernel_sizes, stridess)]
                
                embedding_len = compute_embedding_length(1000, layers_kwargs)
                if embedding_len == -1: continue
                
                final_filters = rng.choice((8, 16, 32))
                
                embedding_size = embedding_len * final_filters
                # DNN models can't have an embedding size larger than 512, so should also CNNs
                if embedding_size > 512: continue
                
                
                starting_filters = rng.choice((256, 128))
                
                possible_filters_reductions = list(product((1, 2), repeat=n_layers-2))
                filters_reductions = possible_filters_reductions[rng.choice(len(possible_filters_reductions))]
                while (starting_filters / np.product(filters_reductions)) < final_filters:
                    filters_reductions = possible_filters_reductions[rng.choice(len(possible_filters_reductions))]
                
                filters = ([starting_filters] +
                            [starting_filters/np.prod(filters_reductions[:i+1]) for i in range(len(filters_reductions))] +
                            [final_filters])
                for i, layer_kwargs in enumerate(layers_kwargs):
                    layer_kwargs["filters"] = filters[i]
                
                dropout_rate = rng.choice((0.05, 0.1))
                
                model_kwargs = {"layers_kwargs": layers_kwargs,
                                "dropout_rate": dropout_rate}
                cnn_kwargs_list.append(model_kwargs)
    
    
    # Generate the list of DNN-type autoencoders.
    FIRST_LAYER_SIZES = (4096, 2048, 1024)
    EMBEDDING_SIZES = (512, 256, 128)
    dnn_kwargs_list = [{"layer_sizes":
        tuple([first_layer_size] +
            [first_layer_size // np.prod(layers_reductions[:layer_id])
            for layer_id in range(1, n_layers + 1)] + [embedding_size]),
        "dropout_rate": dropout_rate}
        for first_layer_size in FIRST_LAYER_SIZES
        for embedding_size in EMBEDDING_SIZES
        for n_layers in range(2, 6)
        for layers_reductions in product((1, 2), repeat=n_layers)
        for dropout_rate in (0.05, 0.1)
        if first_layer_size / np.prod(layers_reductions[:n_layers]) >= embedding_size
    ]


    # These two lines of code are added for easier iteration while keeping track of relevant info
    cnn_kwargs_list = [("cnn", i, model_kwargs) for i, model_kwargs in enumerate(cnn_kwargs_list)]
    dnn_kwargs_list = [("dnn", i, model_kwargs) for i, model_kwargs in enumerate(dnn_kwargs_list)]
    
    return cnn_kwargs_list + dnn_kwargs_list


def compute_input_shapes(experiment_series_dir: Path) -> Tuple[Tuple[int]]:
    """Generate the two training datasets for the two types of models. This is only needed to establish
    what the input shape of the two models is (which is required for building them).

    Args:
        experiment_series_dir (Path): path where the experiments' results can be loaded from 
                to compute the input shapes for the models
    Returns:
        Tuple[int]: two input shape tuples for the two types of model being used.
    """
    cnn_dataset = create_dataset(experiment_series_dir, model_type="cnn", downsampling=downsampling)
    cnn_dataset = cnn_dataset.shuffle(buffer_size=20_000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    dnn_dataset = create_dataset(experiment_series_dir, model_type="dnn", downsampling=downsampling)
    dnn_dataset = dnn_dataset.shuffle(buffer_size=20_000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

    
    for data in cnn_dataset:
        cnn_input_shape = data[0].shape[1:]
        break
    
    for data in dnn_dataset:
        dnn_input_shape = data[0].shape[1:]
        break
    
    return cnn_input_shape, dnn_input_shape


def compute_n_params_distributions(model_kwargs_list: List[Tuple[str, int, ModelParams]], dnn_input_shape: Tuple[int], cnn_input_shape: Tuple[int]) -> Tuple[np.ndarray]:

    # Iterate through all possible models and determine how many params each one has.
    # This is of interest to compare the distribution of model complexity between the CNNs and the DNNs.
    cnn_n_params = []
    dnn_n_params = []
    for model_kwargs in model_kwargs_list:
        model_type = model_kwargs[0]
        if model_type == "cnn":
            model = get_cnn_autoencoder_model(dnn_input_shape, **model_kwargs[2], sigmoid=False)
        else:
            model = get_dnn_autoencoder_model(cnn_input_shape, **model_kwargs[2], sigmoid=False)
        
        n_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables()])
        
        if model_type == "cnn":
            cnn_n_params.append(n_params)
        else:
            dnn_n_params.append(n_params)
        
        tf.keras.backend.clear_session()
    
    return np.array(cnn_n_params), np.array(dnn_n_params)


def main(_) -> None:
    logging.set_verbosity(logging.INFO)
    
    root_path = Path(FLAGS.root_dir), 
    
    model_kwargs_list = generate_model_kwargs_lists()
    
    compute_input_shapes(experiment_series_dir=Path(FLAGS.experiment_series_dir))
    
    cnn_n_params, dnn_n_params = compute_n_params_distributions(model_kwargs_list=model_kwargs_list)
    
    # Saving the two distribution of model complexities that have been computed for analysis
    np.save(root_path.joinpath("cnn_n_params.npy"), np.array(cnn_n_params))
    np.save(root_path.joinpath("dnn_n_params.npy"), np.array(dnn_n_params))


    # Saving the two lists of model params. These will be iterate upon when exploring the possible
    # autoencoder choices
    with open(root_path.joinpath('model_kwargs_list.pickle'), 'wb') as handle:
        pickle.dump(model_kwargs_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    flags.DEFINE_string('root_dir', None,
                        'Root directory for writing results of the metrics.')
    flags.DEFINE_string('experiment_series_dir', None,
                        'Name to identify the experiment series')
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('experiment_series_dir')

    app.run(main)


