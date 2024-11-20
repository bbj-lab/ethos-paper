#!/bin/bash -l
#SBATCH --job-name=ethos_train
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --output=slurm/out/ethos_train.log
#SBATCH --error=slurm/out/ethos_train.err

export OMP_NUM_THREADS=20

case $1 in
mimic | 1)
  dataset=mimic
  data_path=mimic_train_timelines_p241015.hdf5
  vocab_path=mimic_vocab_t4367.pkl
  val_frac=0.04
  ;;
*)
  echo "Wrong experiment number: '$1', available are: 'mimic'"
  exit 1
  ;;
esac

source /home/${USER}/.bashrc
mamba activate ethos

datasets_dir=/gpfs/data/bbj-lab/users/eddie/ethos-paper/ethos/data/tokenized_datasets
data_path=${datasets_dir}/${data_path}
vocab_path=${datasets_dir}/${vocab_path}

BATCH_SIZE=32
BLOCK_SIZE=2048
N_LAYER=6
N_HEAD=12
N_EMBD=768
DROPOUT=0.3
LR=0.0006
MIN_LR=0.00001
gpu_num=8

model_name="layer_${N_LAYER}_batch_${BATCH_SIZE}_do_${DROPOUT}"

torchrun --no_python --standalone --nproc_per_node=$gpu_num ethos train \
  --data_train $data_path \
  --val_frac $val_frac \
  --vocab $vocab_path \
  --batch_size $BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --n_layer $N_LAYER \
  --n_head $N_HEAD \
  --n_embd $N_EMBD \
  --dropout $DROPOUT \
  --lr $LR \
  --min_lr $MIN_LR \
  --log_interval 5 \
  --eval_interval 1000 \
  --gradient_accumulation_steps $gpu_num \
  --max_iters 1000000 \
  --lr_decay_iters 50000 \
  --eval_iters 50 \
  --ctx_no_grad \
  --out_dir "out/${dataset}_${model_name}" \
  --wandb_log \
  --wandb_project "divergence" \
  --wandb_run_name "base_v2.2_${model_name}_resume" \
  --resume \
  --resume_model "recent_model_800000.pt"
