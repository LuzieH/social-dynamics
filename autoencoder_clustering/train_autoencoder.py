import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from utils import create_dataset, get_dnn_autoencoder_model, get_cnn_autoencoder_model

dnn_autoencoder_params = {"layer_sizes": (4096, 1024, 512, 512), "dropout_rate": 0.2}
cnn_autoencoder_params = {
    "layers_kwargs": [{
        "kernel_size": 4,
        "strides": 2,
        "filters": 64
    }, {
        "kernel_size": 7,
        "strides": 2,
        "filters": 128
    }, {
        "kernel_size": 9,
        "strides": 2,
        "filters": 256
    }, {
        "kernel_size": 5,
        "strides": 2,
        "filters": 128
    }, {
        "kernel_size": 7,
        "strides": 2,
        "filters": 64
    }, {
        "kernel_size": 7,
        "strides": 1,
        "filters": 32
    }],
    "dropout_rate": 0.2
}

ROOT_PATH = Path("C:/Users/maler/Federico/Lavoro/ZIB/autoencoder_clustering")
model_type = "cnn"
downsampling = 4

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    experiments_folder = Path("C:/Users/maler/Federico/Lavoro/ZIB/experiments_results")
    experiment_series_folder = experiments_folder.joinpath(
        "2_opt-h_luzie-alpha_beta_gamma_delta_expl-0.0001t")

    dataset = create_dataset(experiment_series_folder, model_type=model_type, downsampling=downsampling)

    for data in dataset:
        input_shape = data[0].shape[1:]
        break
    
    if model_type == "cnn":
        model = get_cnn_autoencoder_model(input_shape, **cnn_autoencoder_params, sigmoid=False)
    else:
        model = get_dnn_autoencoder_model(input_shape, **dnn_autoencoder_params, sigmoid=False)
    
    print("\n\n\n")
    model.summary()
    print("\n\n\n")
    model_hist = model.fit(dataset, epochs=20)
    model_path = ROOT_PATH.joinpath("autoencoder_model", "model.h5")
    model.save(model_path)
    history_path = ROOT_PATH.joinpath("model_history")
    if not history_path.exists():
        history_path.mkdir()
    np.save(history_path.joinpath("history.npy"), model_hist.history)
    #autoencoder = Model(model.input, model.get_layer('embedding').output)
