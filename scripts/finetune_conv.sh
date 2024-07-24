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
WIDTH_MULTIPLIER=$2
DATASET=$3
CROP=$4
OPTIMIZER=$5
LR=$6
DECAY_SCHEDULE=$7
NUM_EPOCHS=$8
WARMUP_EPOCHS=$9
SEED=${10}
SEED_OG=${11}

python main.py --job_type train --config configs/finetune.py:${MODEL},${DATASET},${OPTIMIZER},${DECAY_SCHEDULE} \
      --config.${DATASET}.pp.crop ${CROP} \
      --config.seed=${SEED} \
      --config.width_multiplier=${WIDTH_MULTIPLIER} \
      --config.optimizer.learning_rate ${LR} \
      --config.training_schedule.num_epochs ${NUM_EPOCHS} \
      --config.training_schedule.warmup_epochs ${WARMUP_EPOCHS} \
      --config.pretrained_dir ${PWD}/experiments/${MODEL}x${WIDTH_MULTIPLIER}/scratch/imagenet2012_96/sgd/cosine_1e-1/seed_${SEED_OG} \
      --expdir ${PWD}/experiments/${MODEL}x${WIDTH_MULTIPLIER}/finetuned_imagenet2012_${CROP}_${SEED_OG}/${DATASET}_${CROP}/${OPTIMIZER}/epochs_${NUM_EPOCHS}_${WARMUP_EPOCHS}/${DECAY_SCHEDULE}_${LR}/seed_${SEED}  \

