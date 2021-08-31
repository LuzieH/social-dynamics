import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

from typing import Dict, List, Optional, Tuple

MODEL_INPUT_TYPES = ["cnn-complete", "cnn-cut", "dnn-complete", "dnn-cut", "dnn-batched_complete"]


def compute_embedding_length(time_series_length: int, layers_kwargs: List[Dict[str, int]]) -> int:
    current_len = time_series_length
    for layer_kwargs in layers_kwargs:
        if current_len < layer_kwargs["kernel_size"]:
            return -1

        current_len = (current_len - layer_kwargs["kernel_size"]) / layer_kwargs["strides"] + 1
        if current_len % 1:
            return -1

    return current_len


def create_dataset(series_dir: Path, downsampling: int,
                   model_input_type: str) -> Tuple[tf.data.Dataset, int, int]:
    """Creates a tensorflow.data.Dataset object to be used for training of the autoencoder.

    Args:
        series_dir (Path): Directory where the results for all experiments in the series can be found.
                    These will be used as samples to train the autoencoder.
        downsampling (int): Downsampling factor applied to the results.
        model_input_type (str): String in MODEL_INPUT_TYPES. This is necessary since the two kinds of
                    models require different input shapes.

    Raises:
        ValueError: If model_type is not supported.

    Returns:
        Tuple[tf.data.Dataset, int, int]: Dataset to be used for training the autoencoder, and two integers
                    for the n_agents and n_options of the loaded samples.
    """
    if model_input_type not in MODEL_INPUT_TYPES:
        raise ValueError(f"Invalid model_input_type ({model_input_type})argument passed to the function.")

    model_type, input_type = model_input_type.split("-")
    cut_flag = "cut" in input_type

    def load_numpy_file(file_path: tf.Tensor) -> np.ndarray:
        data = np.load(file_path.numpy().decode())[::downsampling]
        if cut_flag:
            data = data[int(data.shape[0] * 0.75):]
        return data.astype(np.float32)

    example_file = next(series_dir.iterdir()).joinpath("StateMetric", "results_t200000.npy")
    shape = tf.TensorShape(load_numpy_file(tf.convert_to_tensor(str(example_file.absolute()))).shape)
    time_steps, n_agents, n_options = shape.as_list()

    # The value of -1 flattens any remaining dimensions.
    model_input_shape = [-1] if (model_type == "dnn" and ('batched' not in input_type)) else [time_steps, -1]

    def data_preprocessing(exp_data: tf.Tensor) -> tf.Tensor:
        tensor = tf.reshape(exp_data, model_input_shape)
        return tensor, tensor

    def data_pipeline(file_path: str) -> tf.Tensor:
        [exp_data,] = tf.py_function(load_numpy_file, [file_path], [tf.float32,])
        exp_data.set_shape(shape)
        inputs, outputs = data_preprocessing(exp_data)
        return inputs, outputs

    file_pattern = str(series_dir) + "/*/StateMetric/results_t200000.npy"
    dataset = tf.data.Dataset.list_files(file_pattern=file_pattern, shuffle=False)
    dataset = dataset.map(data_pipeline, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    return dataset, n_agents, n_options


def load_all_datasets(series_dir: Path, downsampling: int) -> Tuple[Dict[str, tf.data.Dataset], int, int]:
    """Calls the create_dataset function for all model_types and input_types to be used.
    Stores the resulting datasets in a dictionary which is returned.

    Args:
        series_dir (Path): Directory where the results for all the experiments in the series can be found.
                    These will be used as samples to train the autoencoder.
        downsampling (int): Downsampling factor applied to the results.

    Returns:
        Tuple[Dict[str, tf.data.Dataset], int, int]: Dictionary storing all the datasets loaded with their
                    unique keys, and two integers for the n_agents and n_options of the loaded samples.
    """

    datasets = dict()
    for model_input_type in MODEL_INPUT_TYPES:
        dataset, n_agents, n_options = create_dataset(series_dir=series_dir,
                                                      downsampling=downsampling,
                                                      model_input_type=model_input_type)
        datasets[model_input_type] = dataset

    return datasets, n_agents, n_options


def determine_input_shapes(datasets: Dict[str, tf.data.Dataset]) -> Dict[str, Tuple[int]]:
    """Determines the input shape of all the datasets in the dictionary of datasets.
    
    Note that the Datasets MUST NOT have been batched.

    Args:
        datasets (Dict[str, tf.data.Dataset]): Dictionary containing all the datasets for which the 
                    input_shape must be determined

    Returns:
        Dict[str, Tuple[int]]: Dictionary matching the input's structure and having the input shape
                    tuples as values.
    """
    input_shapes = dict()
    for dataset in datasets:
        for data in datasets[dataset]:
            input_shapes[dataset] = data[0].shape
            break

    return input_shapes


def get_dnn_autoencoder_model(input_shape: int,
                              layer_sizes: Tuple[int],
                              dropout_rate: float,
                              sigmoid=False) -> Model:
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
    opt_m1 = Adam(learning_rate=0.00001)
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
    opt_m1 = Adam(learning_rate=0.00001)
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


def plot_preds(axes: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n_agents: int,
               n_options: int) -> None:
    """Plots the given predictions with their corresponding ground-truth trajectories.

    Args:
        axes (np.ndarray): np.ndarray containing all the axes for the subplots that will be used.
                    This is expected to be analogous to the output of a call as follows:
                        fig, axes = plt.subplots(m, n)
        y_true (np.ndarray): Array of ground-truth trajectories to be plotted.
        y_pred (np.ndarray): Array of autoencoder predictions for the given trajectories.
        n_agents (int): Number of agents in the given experiments. Necessary to reshape.
        n_options (int): Number of options in the given experiments. Necessary to reshape.
    """
    n = y_true.shape[0]

    n_timesteps = int(y_true.size / (n * n_agents * n_options))
    y_true = np.reshape(y_true, (n, n_timesteps, n_agents, n_options))
    y_pred = np.reshape(y_pred, (n, n_timesteps, n_agents, n_options))

    for i in range(n):
        for option in range(n_options):
            ax = axes[i * 2 + option, 0]
            for agent in range(n_agents):
                ax.plot(y_true[i][:, agent, option], label=str(agent))
        for option in range(n_options):
            ax = axes[i * 2 + option, 1]
            for agent in range(n_agents):
                ax.plot(y_pred[i][:, agent, option], label=str(agent))


def select_predictions(mode: str,
                       n_to_sample: int,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       clusters: Optional[np.ndarray] = None,
                       selected_cluster: Optional[int] = None,
                       mses: Optional[np.ndarray] = None,
                       start: Optional[float] = None,
                       end: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    if mode == 'clusters':
        indeces = rng.choice(np.argwhere(clusters == selected_cluster).flatten(),
                             size=n_to_sample,
                             replace=False)
    elif mode == 'mse':
        indeces = rng.choice(np.argwhere((start <= mses) & (mses <= end)).flatten(),
                             size=n_to_sample,
                             replace=False)
    elif mode == 'random':
        indeces = rng.choice(np.arange(y_true.shape[0]), size=n_to_sample, replace=False)
    elif mode == 'worst':
        indeces = np.argsort(mses)[-n_to_sample:]
    else:
        raise ValueError("mode parameter expected to be in ['clusters', 'mse', 'random', 'worst']")

    return y_true[indeces], y_pred[indeces]


def generate_prediction_plots(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              mses: np.ndarray,
                              n_agents: int,
                              n_options: int,
                              save_path: Optional[Path] = None) -> None:
    """Creates and saves in a target folder the images for the autoencoder prediction vs truth.
    This is done for 10 random samples from the dataset and for the worst performing 10 samples (measured 
    using mse).

    Args:
        y_true (np.ndarray): True samples that were fed to the model. First two dimensions are
                    expected to be of size n_samples, n_timesteps
        y_pred (np.ndarray): Predictions of the model for these true samples. First two dimensions are
                    expected to be of size n_samples, n_timesteps
        mses (np.ndarray): Mean Squared Error computed for every sample.
        n_agents (int): Number of agents in the input experiments; used to reconstruct the shape
                    of the samples together with n_timesteps and n_options.
        n_options (int): Number of options in the input experiments; used to reconstruct the shape
                    of the samples together with n_timesteps and n_agents.
        save_path (Optional[Path]): Path to save the generated plots. If not provided, plots
                    are shown instead.
    """
    n_to_plot = 10

    fig, axes = plt.subplots(n_to_plot * n_options, 2, figsize=(20, 60))
    trues, preds = select_predictions(mode='random', n_to_sample=n_to_plot, y_true=y_true, y_pred=y_pred)
    plot_preds(axes=axes, y_true=trues, y_pred=preds, n_agents=n_agents, n_options=n_options)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path.joinpath("random_predictions.png"), dpi=150)

    fig, axes = plt.subplots(n_to_plot * n_options, 2, figsize=(20, 60))
    trues, preds = select_predictions(mode='worst',
                                      n_to_sample=n_to_plot,
                                      y_true=y_true,
                                      y_pred=y_pred,
                                      mses=mses)
    plot_preds(axes=axes, y_true=trues, y_pred=preds, n_agents=n_agents, n_options=n_options)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path.joinpath("worst_predictions.png"), dpi=150)
    else:
        plt.show()

    plt.close('all')
