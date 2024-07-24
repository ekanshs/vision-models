#!/bin/bash
#SBATCH -J train_task
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --partition=a40
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=m2


module load anaconda/3.9
module load cuda11.8+cudnn8.9.6
export PYTHONPATH=$HOME/condaenvs/jax-0.4.23:$PYTHONPATH

cd ..

MODEL=$1
DATASET=$2
OPTIMIZER=$3
LR=$4
DECAY_SCHEDULE=$5
NUM_EPOCHS=$6
WARMUP_EPOCHS=$7
SEED=$


python main.py --job_type train --config configs/finetune.py:${MODEL},${DATASET},${OPTIMIZER},${DECAY_SCHEDULE} \
      --config.seed=${SEED} \
      --config.optimizer.learning_rate ${LR} \
      --config.training_schedule.num_epochs ${NUM_EPOCHS} \
      --config.training_schedule.warmup_epochs ${WARMUP_EPOCHS} \ 
      --expdir ${PWD}/experiments/${MODEL}/finetuned_openai/${DATASET}/${OPTIMIZER}/epochs_${NUM_EPOCHS}_${WARMUP_EPOCHS}/${DECAY_SCHEDULE}_decay/peak_lr_${LR}/seed_${SEED} 
