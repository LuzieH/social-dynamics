import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict



def load_metrics(experiment_dir: str) -> Dict[str, np.ndarray]:
    metrics_results = dict()
    for metric in os.listdir(experiment_dir):
        folder = os.path.join(experiment_dir, metric)
        # Sorting files by modification date. This is done because under lexicographic order
        # results at t = 1000 will come before those at t = 200, leading to incorrect concatenation.
        files = sorted(Path(folder).iterdir(), key=os.path.getmtime)
        results = [np.load(file) for file in files]
        metrics_results[metric] = np.concatenate(results, axis=0)
    
    return metrics_results


