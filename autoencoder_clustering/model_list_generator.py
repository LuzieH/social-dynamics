from absl import app
from absl import flags
from absl import logging

from itertools import product
import numpy as np
from pathlib import Path
import pickle

from social_dynamics.autoencoder_utils import load_all_datasets, determine_input_shapes
from social_dynamics.autoencoder_utils import compute_embedding_length
from social_dynamics.autoencoder_utils import get_dnn_autoencoder_model, get_cnn_autoencoder_model

import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

downsampling = 4
DnnModelKwargs = Dict[str, Union[Tuple[int], float]]
CnnModelKwargs = Dict[str, Union[List[Dict[str, int]], float]]
ModelKwargs = Union[CnnModelKwargs, DnnModelKwargs]


def generate_models_kwargs() -> Dict[str, List[ModelKwargs]]:
    """Generates the kwargs for the autoencoder models to be trained.

    Returns:
        Dict[str, List[ModelParams]]: Dictionary where each key identifies a model_type-input_type combination
                and each value is a list of all model kwargs for the models of that type.
    """
    rng = np.random.default_rng(42)
    models_kwargs = dict()

    # CNN-type autoencoders.
    def generate_cnn_kwargs_list(time_series_length: int) -> List[CnnModelKwargs]:
        """Generates a list of model kwargs for CNN-type autoencoders. These models are valid for time-series
        of the given length.
        
        This is more complex than for the DNNs because the space of hyperparameters is much much larger
        and not all possible choices are allowed. What is allowed is determined via compute_embedding_length()

        Args:
            time_series_length (int): Length of the time series that will be fed in input to the 
            autoencoder.

        Returns:
            List[ModelParams]: List of kwargs for all the models selected.
        """
        
        def generate_n_filters_sequence(starting_filters: int, final_filters: int) -> List[int]:
            possible_filters_reductions = list(product((1, 2), repeat=n_layers - 2))
            filters_reductions = possible_filters_reductions[rng.choice(
                len(possible_filters_reductions))]
            while (starting_filters / np.product(filters_reductions)) < final_filters:
                filters_reductions = possible_filters_reductions[rng.choice(
                    len(possible_filters_reductions))]

            filters = ([starting_filters] +
                       [int(starting_filters / np.prod(filters_reductions[:i + 1]))
                        for i in range(len(filters_reductions))
                        ] +
                       [final_filters])

            return filters
        
        cnn_kwargs_list = []
        for n_layers in range(6, 10):
            # This n_samples value has been empirically determined to make ti so that we get ~200 models
            # for every n_layer. This makes the total number of CNNs comparable to the DNNs.
            n_samples = int(512 / 2.2**(n_layers - 6))
            for stridess in tqdm(list(product((1, 2), repeat=n_layers))):
                for kernel_sizes in rng.choice(np.arange(3, 10), (n_samples, n_layers)):
                    layers_kwargs = [{
                        "kernel_size": int(kernel_size),
                        "strides": int(strides)
                    } for kernel_size, strides in zip(kernel_sizes, stridess)]

                    embedding_len = compute_embedding_length(time_series_length=time_series_length,
                                                             layers_kwargs=layers_kwargs)
                    if embedding_len == -1:
                        continue

                    final_filters = rng.choice((8, 16, 32))

                    embedding_size = embedding_len * final_filters
                    # DNN models can't have an embedding size larger than 512, so should also CNNs
                    if (512 < embedding_size):
                        continue

                    starting_filters = rng.choice((256, 128))

                    filters = generate_n_filters_sequence(starting_filters=starting_filters,
                                                          final_filters=final_filters)
                    
                    for i, layer_kwargs in enumerate(layers_kwargs):
                        layer_kwargs["filters"] = int(filters[i])

                    dropout_rate = rng.choice((0.05, 0.1))

                    model_kwargs = {"layers_kwargs": layers_kwargs, "dropout_rate": dropout_rate}
                    cnn_kwargs_list.append(model_kwargs)

        return cnn_kwargs_list

    models_kwargs["cnn-complete"] = generate_cnn_kwargs_list(1000)
    models_kwargs["cnn-cut"] = generate_cnn_kwargs_list(250)

    # DNN-type autoencoders.
    FIRST_LAYER_SIZES = (4096, 2048, 1024)
    EMBEDDING_SIZES = (512, 256, 128)
    dnn_kwargs = [{
        "layer_sizes":
            tuple([first_layer_size] + [
                first_layer_size // np.prod(layers_reductions[:layer_id])
                for layer_id in range(1, n_layers + 1)
            ] + [embedding_size]),
        "dropout_rate":
            dropout_rate
    }
                  for first_layer_size in FIRST_LAYER_SIZES
                  for embedding_size in EMBEDDING_SIZES
                  for n_layers in range(2, 6)
                  for layers_reductions in product((1, 2), repeat=n_layers)
                  for dropout_rate in (0.05, 0.1)
                  if (first_layer_size / np.prod(layers_reductions[:n_layers])) >= embedding_size]

    models_kwargs["dnn-complete"] = models_kwargs["dnn-cut"] = dnn_kwargs

    return models_kwargs


def compute_n_params_distributions(models_kwargs: Dict[str, List[ModelKwargs]],
                                   input_shapes: Dict[str, Tuple[int]]) -> Dict[str, np.ndarray]:
    """Iterates through all possible models of all possible types to determine how many parameters
    each one has. This is of interest to compare the distribution of model complexity between
    the CNNs and the DNNs.

    Args:
        models_kwargs (Dict[str, List[ModelKwargs]]): Dictionary where each key identifies a
                model_type-input_type combination and each value is a list of all model kwargs
                for the models of that type.
        input_shapes (Dict[str, Tuple[int]]): Dictionary of input shapes with the same keys
                as models_kwargs

    Returns:
        Dict[str, np.ndarray]: Dictionary with the same keys as the two inputs and storing the results
                in np.ndarray objects.
    """
    n_params = dict()
    for key in models_kwargs:
        model_type, input_type = key.split("-")
        input_shape = input_shapes[key]
        key_n_params = []
        for model_kwargs in tqdm(models_kwargs[key]):
            if model_type == "cnn":
                model = get_cnn_autoencoder_model(input_shape, **model_kwargs, sigmoid=False)
            else:
                model = get_dnn_autoencoder_model(input_shape, **model_kwargs, sigmoid=False)

            model_n_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

            key_n_params.append(model_n_params)

            tf.keras.backend.clear_session()

        n_params[key] = np.array(key_n_params)

    return n_params


def main(_) -> None:
    logging.set_verbosity(logging.INFO)

    root_path = Path(FLAGS.root_dir)

    print("\n\nGENERATING MODELS KWARGS\n\n")
    models_kwargs = generate_models_kwargs()

    print("\n\nDETERMINING INPUT SHAPES\n\n")
    input_shapes = determine_input_shapes(
        load_all_datasets(series_dir=root_path.joinpath(FLAGS.series_dir), downsampling=downsampling))

    print("\n\nCOMPUTING N_PARAMS DISTRIBUTIONS\n\n")
    models_n_params = compute_n_params_distributions(models_kwargs=models_kwargs, input_shapes=input_shapes)

    # Saving the distributions of model complexities that have been computed for analysis
    for key in models_n_params:
        np.save(root_path.joinpath(key + ".npy"), models_n_params[key])

    # Saving the dict of model params. This will be iterated upon when exploring the possible
    # autoencoder choices
    with open(root_path.joinpath('models_kwargs.pickle'), 'wb') as handle:
        pickle.dump(models_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    flags.DEFINE_string('root_dir', None, 'Root directory for writing results of the metrics.')
    flags.DEFINE_string('series_dir', None, 'Name to identify the experiment series')
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('series_dir')

    app.run(main)
