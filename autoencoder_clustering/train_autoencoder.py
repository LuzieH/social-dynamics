import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from utils import create_dataset, get_autoencoder_model





if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
    experiments_folder = Path("C:/Users/maler/Federico/Lavoro/ZIB/experiments_results")
    experiment_series_folder = experiments_folder.joinpath("2_opt-h_luzie-alpha_beta_gamma_delta_expl-0.0001t")


    dataset = create_dataset(experiment_series_folder)

    for data in dataset:
        input_shape = data[0].shape[1]
        break

    model = get_autoencoder_model(input_shape, layer_sizes=(2048, 512, 128, 32), sigmoid=False)
    model_hist = model.fit(dataset, epochs=10)
    model_path = Path("autoencoder_model/model.h5")
    model.save(model_path)
    history_path = Path("model_history")
    history_path.mkdir()
    np.save(history_path.joinpath("history.npy"), model_hist.history)
    #autoencoder = Model(model.input, model.get_layer('embedding').output)

