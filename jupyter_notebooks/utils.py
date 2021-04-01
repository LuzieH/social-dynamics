import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Optional
import ternary



def load_metrics(experiment_dir: str) -> Dict[str, np.ndarray]:
    metrics_results = dict()
    for metric in os.listdir(experiment_dir):
        if metric == 'initial_random_state.npy': continue
        folder = os.path.join(experiment_dir, metric)
        # Sorting files by modification date. This is done because under lexicographic order
        # results at t = 1000 will come before those at t = 200, leading to incorrect concatenation.
        files = sorted(Path(folder).iterdir(), key=os.path.getmtime)
        results = [np.load(file) for file in files]
        metrics_results[metric] = np.concatenate(results, axis=0)
    
    return metrics_results


def plot_agents_simplex(options: np.ndarray, filename: Optional[str] = None) -> None:
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
    
    if filename is not None:
        plt.savefig(filename)
        plt.close()
        return
    tax.show()


def plot_agents_option(agents: np.ndarray, filename: Optional[str] = None) -> None:
    """
    Plots the values across time for all options on different subplots.
    
    Args:
        agents: A numpy array of values to be plotted. Shape=(n_time_steps, n_agents, n_options)
        filename: Path to save the figure in instead of plotting it.
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
    
    if filename is not None:
        plt.savefig(filename)
        plt.close()
        return
    
    plt.show()

