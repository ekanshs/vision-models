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

python main.py --job_type finetune --config configs/ft_ta.py:$1,$2,$3,$4 --job_type=finetune --expdir ${PWD}/experiments/$1/$2/$3/ta_$4/seed_$5 --config.seed=$5 
