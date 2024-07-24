#!/bin/bash
#SBATCH -J train_task
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --partition=a40
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=m


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
SEED=$8

python main.py --job_type train --config configs/train_from_scratch.py:${MODEL},${DATASET},${OPTIMIZER},${DECAY_SCHEDULE} \
       --config.${DATASET}.pp.crop ${CROP} \
       --config.seed=${SEED} \
       --config.width_multiplier=${WIDTH_MULTIPLIER} \
       --config.optimizer.learning_rate ${LR} \
       --expdir ${PWD}/experiments/${MODEL}x${WIDTH_MULTIPLIER}/scratch/${DATASET}_${CROP}/${OPTIMIZER}/${DECAY_SCHEDULE}_decay/peak_lr_${LR}/seed_${SEED} 

