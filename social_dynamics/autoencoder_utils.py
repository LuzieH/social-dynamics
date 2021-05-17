from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

from typing import Callable, Dict, List, Tuple


def check_cnn_autoencoder_shapes(time_series_length: int, layers_kwargs: List[Dict[str, int]]) -> bool:
    current_len = time_series_length
    for layer_kwargs in layers_kwargs:
        current_len = (current_len - layer_kwargs["kernel_size"]) / layer_kwargs["strides"] + 1
        if current_len % 1:
            return False
    
    return True


def compute_embedding_size(time_series_length: int, layers_kwargs: List[Dict[str, int]]) -> bool:
    current_len = time_series_length
    for layer_kwargs in layers_kwargs:
        current_len = (current_len - layer_kwargs["kernel_size"]) / layer_kwargs["strides"] + 1
    
    return current_len * layers_kwargs[-1]["filters"]


def load_numpy_file(file_path: str, downsampling: int) -> np.ndarray:
    data = np.load(file_path.numpy().decode())[::downsampling]
    return data.astype(np.float32)


def dnn_data_preprocessing(exp_data: tf.Tensor) -> tf.Tensor:
    tensor = tf.reshape(exp_data, [-1])
    return tensor, tensor


def dnn_data_pipeline(file_path: str, shape: tf.TensorShape, load_func: Callable[[str, int],
                                                                                 np.ndarray]) -> tf.Tensor:
    [exp_data,] = tf.py_function(load_func, [file_path], [tf.float32,])
    exp_data.set_shape(shape)
    inputs, outputs = dnn_data_preprocessing(exp_data)
    return inputs, outputs


def cnn_data_preprocessing(exp_data: tf.Tensor) -> tf.Tensor:
    tensor = tf.reshape(exp_data, [tf.shape(exp_data)[0], -1])
    return tensor, tensor


def cnn_data_pipeline(file_path: str, shape: tf.TensorShape, load_func: Callable[[str, int],
                                                                                 np.ndarray]) -> tf.Tensor:
    [exp_data,] = tf.py_function(load_func, [file_path], [tf.float32,])
    exp_data.set_shape(shape)
    inputs, outputs = cnn_data_preprocessing(exp_data)
    return inputs, outputs


def create_dataset(series_dir: Path, downsampling: int, model_type: str) -> tf.data.Dataset:
    if model_type not in ["cnn", "dnn"]:
        raise ValueError(f"Invalid model_type ({model_type})argument passed to the function.")
    example_file = next(series_dir.iterdir()).joinpath("StateMetric", "results_t200000.npy")
    shape = tf.TensorShape(np.load(example_file)[::downsampling].shape)
    data_pipeline_func = partial(cnn_data_pipeline if model_type == "cnn" else dnn_data_pipeline,
                                 shape=shape,
                                 load_func=partial(load_numpy_file, downsampling=downsampling))
    file_pattern = str(series_dir) + "/*/StateMetric/results_t200000.npy"
    dataset = tf.data.Dataset.list_files(file_pattern=file_pattern, shuffle=False)
    dataset = dataset.map(data_pipeline_func, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    return dataset


def get_dnn_autoencoder_model(input_shape: int, layer_sizes: Tuple[int], dropout_rate: float, sigmoid=False):
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
                              sigmoid=False):
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
