#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:0
#SBATCH --job-name=InstallEnv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm/out/install_env_%A.txt

module purge
module load 2021
module load Anaconda3/2021.05

cd $HOME/CoT-XAI-HateSpeechDetection
conda env create -f environment.yml

echo "Job finished fully."
