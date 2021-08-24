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

## This folder would also be created by the python code below, but this would raise an error and crash all the tasks
## due to the slow time of access to the filesystem and to the fact that they all try to create the same folder 
## at the same time. Therefore I instead create the folder a single time here before starting all the tasks in parallel.
mkdir -p experiments_results/2_opt-h_luzie-alpha_beta_gamma_delta_expl-0.0001t

## The --ntasks flag decides how many different instances of the same command should be run in parallel.
## In this case the command is the python code to be executed.
## The -o flag tells SLURM where to write the logs (i.e. the terminal output) for the python file's execution.
## Note that the output location given by the `#SBATCH -o` above is for the output of the overall SLURM job
## Which for example logs when, if, why tasks have completed/stopped. This output location is instead for the
## single task (therefore there are actually ntasks output files being generated here).
srun --ntasks=60 -o /home/htc/fmalerba/logs/%x_%J_%t.out python code/social-dynamics/run_experiment_series.py \
--root_dir="experiments_results" \
--gin_files="social-dynamics/configs/2_options_homogenous_luzie.gin" \
--series_name="2_opt-h_luzie-alpha_beta_gamma_delta_expl-0.0001t" \
--gin_bindings="time_interval=0.0001"
wait