#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=LaMini-GPT-124M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm/out/LaMini-GPT-124M_%A.txt

module purge
module load 2021
module load Anaconda3/2021.05

source activate CoT-XAI-HateSpeechDetection

cd $HOME/CoT-XAI-HateSpeechDetection
srun python3 src/eval.py --model LaMini-GPT-124M

echo "Job finished fully."