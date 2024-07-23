#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --job-name=vscode
#SBATCH --output=logs/vscode_%j.log
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=a40
#SBATCH --qos=m

date;hostname;pwd

cd $SLURM_SUBMIT_DIR
module load vscode-server
module load anaconda/3.9
module load cuda11.8+cudnn8.9.6
export PYTHONPATH=$HOME/condaenvs/jax-0.4.23:$PYTHONPATH

code-server serve-local --host $(hostname --fqdn) --port 12000 --accept-server-license-terms --disable-telemetry
