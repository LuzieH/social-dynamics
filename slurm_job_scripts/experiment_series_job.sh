#!/bin/bash
#SBATCH -p big # partition
#SBATCH --ntasks=60
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o /home/htc/fmalerba/logs/%x.out
#SBATCH --job-name=experiment_series
#SBATCH --mem=64000 


cd /home/htc/fmalerba/
source /home/htc/fmalerba/venv/bin/activate

mkdir -p experiments_results/2_opt-h_luzie-alpha_beta_gamma_delta_expl-0.0001t

srun --export=ALL --ntasks=60 -o /home/htc/fmalerba/logs/%x_%J_%t.out python code/social-dynamics/run_experiment_series.py \
--root_dir="experiments_results" \
--gin_files="social-dynamics/configs/2_options_homogenous_luzie.gin" \
--series_name="2_opt-h_luzie-alpha_beta_gamma_delta_expl-0.0001t" \
--gin_bindings="time_interval=0.0001"
wait