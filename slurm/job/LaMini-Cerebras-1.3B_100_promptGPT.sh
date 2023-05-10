#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=LaMini-Cerebras-1.3B
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --mem=32000M
#SBATCH --output=slurm/out/LaMini-Cerebras-1.3B_100_promptGPT_%A.txt

module purge
module load 2021
module load Anaconda3/2021.05

source activate CoT-XAI-HateSpeechDetection

cd $HOME/CoT-XAI-HateSpeechDetection
srun python3 src/eval.py --model LaMini-Cerebras-1.3B --show_preds 10 --min_preds 100 --prompt_path ./prompts/promptGPT.txt

echo "Job finished fully."
