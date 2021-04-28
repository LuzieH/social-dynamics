from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras

from typing import Tuple


def load_numpy_file(file_path: str) -> np.ndarray:
    data = np.load(file_path.numpy().decode())
    return data.astype(np.float32)


def data_preprocessing(exp_data: tf.Tensor) -> tf.Tensor:
    return tf.reshape(exp_data[::2], [-1]), tf.reshape(exp_data[::2], [-1])


def load_data(file_path: str, shape: tf.TensorShape) -> tf.Tensor:
    [exp_data,] = tf.py_function(load_numpy_file, [file_path], [tf.float32,])
    inputs, outputs = tf.py_function(data_preprocessing, [exp_data], [tf.float32, tf.float32])
    inputs.set_shape(shape)
    outputs.set_shape(shape)
    return inputs, outputs


def create_dataset(series_dir: Path) -> tf.data.Dataset:
    example_file = [exp_dir for exp_dir in series_dir.iterdir()][0].joinpath("StateMetric", "results_t200000.npy")
    shape = data_preprocessing(np.load(example_file))[0].shape
    load_func = partial(load_data, shape=shape)
    file_pattern = str(series_dir) + "/*/StateMetric/results_t200000.npy"
    dataset = tf.data.Dataset.list_files(file_pattern=file_pattern, shuffle=True)
    dataset = dataset.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    dataset = dataset.batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def get_autoencoder_model(input_shape, layer_sizes: Tuple[int], sigmoid=False):
    model_input = model = Input(shape=(input_shape, ))
    
    for layer_size in layer_sizes[:-1]:
        model = Dense(layer_size, activation='relu')(model)
    
    model = Dense(layer_sizes[-1], activation='relu',name='embedding')(model)
    
    for layer_size in layer_sizes[:-1][::-1]:
        model = Dense(layer_size, activation='relu')(model)

    if sigmoid:
        model = Dense(input_shape, activation='sigmoid')(model)
    else:
        model = Dense(input_shape, activation='linear')(model)
    model = Model(inputs=model_input, outputs=model)
    opt_m1 = Adam(lr=0.001)
    model.compile(optimizer=opt_m1, loss="mean_squared_error", metrics=['mse'])
    return model





