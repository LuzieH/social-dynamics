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

## Necessary for Tensorflow
export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

## The --export=ALL flag tells SLURM to give each task access to all the terminal environment variables
## that this terminal has access to. This is necessary to make sure that the `export` commands above work
## for every single task being spawned.
## The --ntasks flag decides how many different instances of the same command should be run in parallel.
## In this case the command is the python code to be executed.
## The -o flag tells SLURm where to write the logs (i.e. the terminal output) for the python file's execution.
## Note that the output location given by the `#SBATCH -o` above is for the output of the overall SLURM job
## Which for example logs when, if, why tasks have completed/stopped. This output location is instead for the
## single task (therefore there are actually ntasks output files being generated here).
srun --export=ALL --ntasks=1 -o /home/htc/fmalerba/logs/%x_%J_%t.out python code/social-dynamics/autoencoder_clustering/model_list_generator.py \
--root_dir="/scratch/htc/fmalerba/autoencoder_clustering" \
--series_dir="/home/htc/fmalerba/experiments_results/2_opt-h_luzie-alpha_beta_gamma_delta_expl-0.0001t"
wait