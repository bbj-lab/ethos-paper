#!/bin/bash -l
#SBATCH --job-name=split_train_test
#SBATCH --time=8:00:00
#SBATCH --partition=tier2q
#SBATCH --chdir=/gpfs/data/bbj-lab/users/eddie/ethos-paper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=slurm/out/split_train_test.log

source /home/${USER}/.bashrc
mamba activate ethos
python scripts/data_train_test_split.py ethos/data/mimic-iv-2.2
python scripts/convert_csv_to_parquet.py ethos/data/mimic-iv-2.2_Data