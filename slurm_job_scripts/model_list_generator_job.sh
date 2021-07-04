#!/bin/bash
#SBATCH -p big # partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH -o /home/htc/fmalerba/logs/%x.out
##SBATCH --gres=gpu:1
#SBATCH --job-name=model_list_generator_job
#SBATCH --mem=64000 

cd /home/htc/fmalerba/
source /home/htc/fmalerba/venv/bin/activate

export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


srun --export=ALL --ntasks=1 -o /home/htc/fmalerba/logs/%x_%J_%t.out python code/social-dynamics/autoencoder_clustering/model_list_generator.py \
--root_dir="/home/htc/fmalerba/autoencoder_clustering" \
--series_dir="/home/htc/fmalerba/experiments_results/2_opt-h_luzie-alpha_beta_gamma_delta_expl-0.0001t"
wait