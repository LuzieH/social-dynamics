#!/bin/bash
#SBATCH -p gpu # partition
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=3
#SBATCH --time=1-00:00:00
#SBATCH -o /home/htc/fmalerba/logs/%x.out
#SBATCH --gres=gpu:1
#SBATCH --job-name=autoencoder_exploration_job
#SBATCH --mem=128000

cd /home/htc/fmalerba/
source /home/htc/fmalerba/venv/bin/activate

export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

mkdir -p /scratch/htc/fmalerba/autoencoder_clustering/autoencoders_results

srun --export=ALL --ntasks=3 -o /home/htc/fmalerba/logs/%x_%J_%t.out python code/social-dynamics/autoencoder_clustering/autoencoder_exploration.py \
--root_dir="/scratch/htc/fmalerba/autoencoder_clustering" \
--series_dir="/home/htc/fmalerba/experiments_results/2_opt-h_luzie-alpha_beta_gamma_delta_expl-0.0001t" \
--batch_size=10 \
--logging="info"
wait