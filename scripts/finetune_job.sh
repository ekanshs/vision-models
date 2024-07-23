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
DECAY_SCHEDULE=$4
NUM_EPOCHS=$5
WARMUP_EPOCHS=$6
LR=$7
WD=$8
SEED=$9

python main.py --job_type finetune \
       --config configs/finetune.py:${MODEL},${DATASET},${OPTIMIZER},${DECAY_SCHEDULE} \
       --config.seed ${SEED} \
       --config.training_schedule.num_epochs ${NUM_EPOCHS} \
       --config.training_schedule.warmup_epochs ${WARMUP_EPOCHS} \
       --config.optimizer.learning_rate ${LR} \
       --config.optimizer.weight_decay ${WD} \
       --config.optimizer.clip_global_norm 100.0 \
       --config.optimizer.classifier.name adam \
       --config.optimizer.classifier.learning_rate 1e-5 \
       --config.optimizer.classifier.momentum 0.9 \
       --config.optimizer.classifier.weight_decay ${WD} \
       --config.optimizer.classifier.clip_global_norm 1.0 \
       --expdir ${PWD}/experiments/${MODEL}/${DATASET}/${OPTIMIZER}/LR_${LR}_WD_${WD}/${DECAY_SCHEDULE}/${NUM_EPOCHS}_epochs_${WARMUP_EPOCHS}_warmup/seed_${SEED} \


