#!/bin/bash
#SBATCH -J adapt_task
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --partition=a40
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --gres=gpu:4
#SBATCH --qos=m2


module load anaconda/3.9
module load cuda11.8+cudnn8.9.6
export PYTHONPATH=$HOME/condaenvs/jax-0.4.23:$PYTHONPATH

python main.py --job_type conv-train --config configs/conv_adapt.py:ResNet50,imagenet2012,adam,cosine --expdir ${PWD}/experiments/ResNet50/imagenet2012/1/train_encoder --config.seed=0 --config.pretrained_dir experiments/ResNet50/imagenet2012/1/ --config.train_encoder=True --config.optimizer.learning_rate=1e-5 --config.training_schedule.num_epochs=10
