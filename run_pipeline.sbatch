#!/bin/bash

#SBATCH -o ./slurm_logs/output_job_%j.job
#SBATCH -e ./slurm_logs/error_job_%j.job
#SBATCH -J intelex
#SBATCH -c 32 #CPU cores required
#SBATCH -p cpu_p
#SBATCH --mem=256G #Memory required
#SBATCH --qos=cpu_normal
#SBATCH -t 1-00:00:00 
#SBATCH --nice=1000 #Manual priority. Do not change this.

source $HOME/.bashrc
conda activate MLEE
python -m sklearnex.glob patch_sklearn -nv
python main.py
python -m sklearnex.glob unpatch_sklearn -nv
