#!/bin/bash
#SBATCH -J train_task
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --partition=a40
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=m3


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
SEED=$8
PRETRAINED_DIR=$9
PRETRAINED_NAME=${10}


if test -d ${PRETRAINED_DIR}/checkpoint_* ; then 
  echo ${PRETRAINED_DIR}
  echo "pretrained directory exists"; 
  SL_DIRECTORY=${PWD}/experiments/${MODEL}/${PRETRAINED_NAME}/${DATASET}/${OPTIMIZER}/epochs_${NUM_EPOCHS}_${WARMUP_EPOCHS}/${DECAY_SCHEDULE}_${LR}
  EXPDIR=${SL_DIRECTORY}/seed_${SEED}
  PT_FLAG=1
else  
  echo "pretrained directory does not exists; defaulting to openai model"; 
  SL_DIRECTORY=${PWD}/experiments/${MODEL}/openai/${DATASET}/${OPTIMIZER}/epochs_${NUM_EPOCHS}_${WARMUP_EPOCHS}/${DECAY_SCHEDULE}_${LR}
  EXPDIR=${SL_DIRECTORY}/seed_${SEED}
  echo ${EXPDIR}
fi

if test -d $SL_DIRECTORY; then 
  if test -d $EXPDIR; then
    echo "${EXPDIR} directory exists. Loading model from this directory"; 
  else
    ln -s /checkpoint/${USER}/${SLURM_JOB_ID}/ $EXPDIR
  fi
else
  mkdir -p $SL_DIRECTORY
  ln -s /checkpoint/${USER}/${SLURM_JOB_ID}/ $EXPDIR  
fi
if [ PT_FLAG -eq 1 ]; then
  python main.py --job_type train \
      --config configs/finetune.py:${MODEL},${DATASET},${OPTIMIZER},${DECAY_SCHEDULE} \
      --config.seed=${SEED} \
      --config.optimizer.learning_rate ${LR} \
      --config.training_schedule.num_epochs ${NUM_EPOCHS} \
      --config.training_schedule.warmup_epochs ${WARMUP_EPOCHS} \
      --config.pretrained_dir ./${PRETRAINED_DIR} \
      --expdir ${EXPDIR} 
else 
    python main.py --job_type train \
      --config configs/finetune.py:${MODEL},${DATASET},${OPTIMIZER},${DECAY_SCHEDULE} \
      --config.seed=${SEED} \
      --config.optimizer.learning_rate ${LR} \
      --config.training_schedule.num_epochs ${NUM_EPOCHS} \
      --config.training_schedule.warmup_epochs ${WARMUP_EPOCHS} \
      --expdir ${EXPDIR} 
fi