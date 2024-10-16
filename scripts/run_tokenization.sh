#!/bin/bash -l
#SBATCH --job-name=ethos_tokenize
#SBATCH --time=6:00:00
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=slurm/out/ethos_tokenize.log

source /home/${USER}/.bashrc
mamba activate ethos
ethos tokenize- $*