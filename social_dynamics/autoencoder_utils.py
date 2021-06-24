from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

from typing import Dict, List, Tuple


MODEL_TYPES = ["cnn", "dnn"]
INPUT_TYPES = ["complete", "cut"]

def compute_embedding_length(time_series_length: int, layers_kwargs: List[Dict[str, int]]) -> int:
    current_len = time_series_length
    for layer_kwargs in layers_kwargs:
        current_len = (current_len - layer_kwargs["kernel_size"]) / layer_kwargs["strides"] + 1
        if current_len % 1:
            return -1
    
    return current_len


def load_numpy_file(file_path: str, downsampling: int, cut: bool) -> np.ndarray:
    data = np.load(file_path.numpy().decode())[::downsampling]
    if cut:
        data = data[int(data.shape[0]*0.75):]
    return data.astype(np.float32)


def dnn_data_preprocessing(exp_data: tf.Tensor) -> tf.Tensor:
    tensor = tf.reshape(exp_data, [-1])
    return tensor, tensor


def cnn_data_preprocessing(exp_data: tf.Tensor) -> tf.Tensor:
    tensor = tf.reshape(exp_data, [tf.shape(exp_data)[0], -1])
    return tensor, tensor


def create_dataset(series_dir: Path, downsampling: int, model_type: str, cut: bool) -> tf.data.Dataset:
    """Creates a tensorflow.data.Dataset object to be used for training of the autoencoder.

    Args:
        series_dir (Path): Directory where the results for all experiments in the series can be found.
                    These will be used as samples to train the autoencoder.
        downsampling (int): Downsampling factor applied to the results.
        model_type (str): String in MODEL_TYPES. This is necessary since the two kinds of models require
                    different input shapes.
        cut (bool): Boolean for whether to feed the last 25% of an experiment as input instead of
                    the entire run.

    Raises:
        ValueError: If model_type is not supported.

    Returns:
        tf.data.Dataset: Dataset to be used for training the autoencoder.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model_type ({model_type})argument passed to the function.")
    example_file = next(series_dir.iterdir()).joinpath("StateMetric", "results_t200000.npy")
    load_func = partial(load_numpy_file, downsampling=downsampling, cut=cut)
    shape = tf.TensorShape(load_func(example_file).shape)
    
    preprocessing_func = dnn_data_preprocessing if model_type == "dnn" else cnn_data_preprocessing
    def data_pipeline(file_path: str) -> tf.Tensor:
        [exp_data,] = tf.py_function(load_func, [file_path], [tf.float32,])
        exp_data.set_shape(shape)
        inputs, outputs = preprocessing_func(exp_data)
        return inputs, outputs
    
    file_pattern = str(series_dir) + "/*/StateMetric/results_t200000.npy"
    dataset = tf.data.Dataset.list_files(file_pattern=file_pattern, shuffle=False)
    dataset = dataset.map(data_pipeline, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    return dataset


def load_all_datasets(series_dir: Path, downsampling: int) -> Dict[str, tf.data.Dataset]:
    """Calls the create_dataset function for all model_types and input_types to be used.
    Stores the resulting datasets in a dictionary which is returned.

    Args:
        series_dir (Path): Directory where the results for all the experiments in the series can be found.
                    These will be used as samples to train the autoencoder.
        downsampling (int): Downsampling factor applied to the results.

    Returns:
        Dict[str, tf.data.Dataset]: Dictionary storing all the datasets loaded with their unique keys. 
    """
    
    datasets = dict()
    for model_type in MODEL_TYPES:
        for input_type in INPUT_TYPES:
            key = "-".join((model_type, input_type))
            
            dataset = create_dataset(series_dir=series_dir, downsampling=downsampling, model_type=model_type, cut=(input_type == "cut"))
            dataset = dataset.shuffle(buffer_size=20_000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
            datasets[key] = dataset
    
    return datasets


def determine_input_shapes(datasets: Dict[str, tf.data.Dataset]) -> Dict[str, Tuple[int]]:
    input_shapes = dict()
    for dataset in datasets:
        for data in datasets[dataset]:
            input_shapes[dataset] = data[0].shape[1:]
            break
    
    return input_shapes



def get_dnn_autoencoder_model(input_shape: int, layer_sizes: Tuple[int], dropout_rate: float, sigmoid=False) -> Model:
    model_input = model = Input(shape=input_shape)

    for layer_size in layer_sizes[:-1]:
        model = Dense(layer_size, activation='relu')(model)
        model = Dropout(dropout_rate)(model)

    model = Dense(layer_sizes[-1], activation='relu', name='embedding')(model)
    model = Dropout(dropout_rate)(model)

    for layer_size in layer_sizes[:-1][::-1]:
        model = Dense(layer_size, activation='relu')(model)
        model = Dropout(dropout_rate)(model)

    if sigmoid:
        model = Dense(input_shape[0], activation='sigmoid')(model)
    else:
        model = Dense(input_shape[0], activation='linear')(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.00001)
    model.compile(optimizer=opt_m1, loss="mean_squared_error", metrics=['mse'])
    return model


def get_cnn_autoencoder_model(input_shape: Tuple[int],
                              layers_kwargs: List[Dict[str, int]],
                              dropout_rate: float,
                              sigmoid=False) -> Model:
    model_input = model = Input(shape=input_shape)

    for layer_kwargs in layers_kwargs[:-1]:
        model = Conv1D(**layer_kwargs, activation='relu')(model)
        model = SpatialDropout1D(dropout_rate)(model)

    model = Conv1D(**(layers_kwargs[-1]), activation='relu', name='embedding')(model)
    SpatialDropout1D(dropout_rate)(model)

    for layer_kwargs in layers_kwargs[:0:-1]:
        model = Conv1DTranspose(**layer_kwargs, activation='relu')(model)
        model = SpatialDropout1D(dropout_rate)(model)

    if sigmoid:
        model = Conv1DTranspose(kernel_size=layers_kwargs[0]["kernel_size"],
                                strides=layers_kwargs[0]["strides"],
                                filters=input_shape[1],
                                activation='sigmoid')(model)
    else:
        model = Conv1DTranspose(kernel_size=layers_kwargs[0]["kernel_size"],
                                strides=layers_kwargs[0]["strides"],
                                filters=input_shape[1],
                                activation='linear')(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.00001)
    model.compile(optimizer=opt_m1, loss="mean_squared_error", metrics=['mse'])
    return model


def plot_history(h, metric='acc'):
    if metric == 'acc':
        plt.title('Accuracy')
    else:
        plt.title(metric)

    if metric == 'mean_squared_error':
        plt.yscale('log')

    plt.plot(h.history['val_' + metric], label='validation')
    plt.plot(h.history[metric], label='train')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.show()
