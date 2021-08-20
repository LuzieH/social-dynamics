import io
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import ternary
from tqdm import tqdm
from typing import Dict, List, Optional


def load_metrics(experiment_dir: Path) -> Dict[str, np.ndarray]:
    metrics_results = dict()
    for metric in experiment_dir.iterdir():
        if metric.name == 'initial_random_state.npy': continue
        # Sorting files by modification date. This is done because under lexicographic order
        # results at t = 1000 will come before those at t = 200, leading to incorrect concatenation.
        files = sorted(metric.iterdir(), key=os.path.getmtime)
        results = [np.load(file) for file in files]
        metrics_results[metric.name] = np.concatenate(results, axis=0)
    
    return metrics_results


def plot_agents_simplex(options: np.ndarray, save_path: Optional[Path] = None) -> None:
    _, ax = plt.subplots(figsize=(8, 8), dpi=150)
    _, tax = ternary.figure(ax=ax, scale=1.0)
    tax.boundary()
    tax.gridlines(multiple=0.2, color="black")
    tax.set_title("Agent trajectories", fontsize=20)
    fontsize = 12
    offset = 0.14
    tax.left_axis_label("Option 3", fontsize=fontsize, offset=offset)
    tax.right_axis_label("Option 2", fontsize=fontsize, offset=offset)
    tax.bottom_axis_label("Option 1", fontsize=fontsize, offset=offset)
    
    n_agents = options.shape[1]
    for agent in range(n_agents):
        tax.plot(options[:, agent, :], linewidth=1.0, label=str(agent))
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f")
    
    plt.axis('off')
    tax.legend()
    
    if save_path is not None:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        plt.savefig(save_path)
        plt.close()
        return
    tax.show()


def plot_agents_option(agents: np.ndarray, save_path: Optional[Path] = None) -> None:
    """
    Plots the values across time for all options on different subplots.
    
    Args:
        agents: A numpy array of values to be plotted. Shape=(n_time_steps, n_agents, n_options)
        save_path: Path to save the figure in instead of plotting it.
    """
    n_agents, n_options = agents.shape[1:]
    
    plt.figure(figsize=(14, 8))
    for option in range(n_options):
        plt.subplot(n_options, 1, option+1)
        for agent in range(n_agents):
            plt.plot(agents[:, agent, option], label=str(agent))
            plt.legend()
            plt.title("Option "+ str(option+1))

    plt.tight_layout()
    
    if save_path is not None:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        plt.savefig(save_path)
        plt.close()
        return
    
    plt.show()


def generate_gif(save_path: Path, state_metric: np.ndarray) -> None:
    """Takes as input a state_metric result matrix and produces a GIF with the evolution of the 
    of the agents' states across time plotted in the cartesian plane.

    Args:
        save_path (Path): Path to the GIF savefiel
        state_metric (np.ndarray): State metric to be plotted in the GIF. Must be from an experiment
                with at most two options
    """
    if state_metric.shape[2] != 2:
        raise ValueError("Cannot generate GIF for experiments with more than two options")
    
    x_low, y_low, x_high, y_high = *np.min(state_metric, axis=(0,1)), *np.max(state_metric, axis=(0,1))

    images_bufs = []
    for i in tqdm(range(5, state_metric.shape[0]- 5)):
        to_plot = state_metric[i:i+5]
        
        plt.figure()
        for agent in range(state_metric.shape[1]):
            plt.plot(to_plot[:, agent, 0], to_plot[:, agent, 1])
        plt.xlim(x_low, x_high)
        plt.ylim(y_low, y_high)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        images_bufs.append(buf)
        plt.close('all')

    # Build gif
    with imageio.get_writer(save_path, mode='I') as writer:
        for buf in tqdm(images_bufs):
            buf.seek(0)
            image = imageio.imread(buf)
            writer.append_data(image)
            buf.close()


def determine_root_path() -> Path:
    if "fmalerba" in str(Path(".").resolve()):
        return Path("/scratch/htc/fmalerba/")
    elif "maler" in str(Path(".").resolve()):
        return Path("C:/Users/maler/Federico/Lavoro/ZIB/") 
    elif "luziehel" in str(Path(".").resolve()):
        return Path("C:/Users/luziehel/Code/")


def autoencoder_sorting(path: Path) -> int:
    """Function used as a key to sort the autoencoder results' paths.
    Without using this ".../cnn-cut-10/" would come before ".../cnn-cut-2/"

    Args:
        path (Path): Path to an autoencoder's results

    Returns:
        int: integer to be used for sorting, corresponds to the model's id.
    """
    return int(path.name.split("-")[-1])


def load_autoencoder_exploration_results(path: Path, model_input_types: List[str]) -> pd.DataFrame:
    results = pd.DataFrame(columns=["Model-Input Type", "Model ID", "MSE", "N. Params"])
    for model_input_type in model_input_types:
        n_params = np.load(path.joinpath(model_input_type + "-n_params_distribution.npy"))
        for autoenc in sorted(path.joinpath('autoencoders_results').iterdir(), key=autoencoder_sorting):
            if model_input_type not in autoenc.name: continue
            model_id = int(autoenc.name.split("-")[-1])
            mse = np.mean(np.load(autoenc.joinpath("mses.npy")))
            row = {"Model-Input Type": model_input_type,
                   "Model ID": model_id,
                   "MSE": mse,
                   "N. Params": n_params[model_id]}
            
            results = results.append(row, ignore_index=True)
    
    return results

