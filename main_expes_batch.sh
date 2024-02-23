#!/bin/bash

#SBATCH --array=0-185
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --time=5:30:00
#SBATCH --job-name=expes_reco_rf
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray

python3.10 data_extraction_from_rf_experiments.py --expe_id=$SLURM_ARRAY_TASK_ID