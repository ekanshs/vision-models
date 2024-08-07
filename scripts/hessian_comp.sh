#!/bin/bash
#SBATCH -J hessian
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH --time=4:00:00
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
CROP_SIZE=$4
EXPDIR=$5

python main.py --job_type hessian \
       --config configs/hessian.py:${MODEL},${DATASET} \
       --config.width_multiplier ${WIDTH_MULTIPLIER} \
       --config.${DATASET}.pp.crop ${CROP_SIZE} \
       --config.compute_trace=False \
       --expdir ${EXPDIR}
