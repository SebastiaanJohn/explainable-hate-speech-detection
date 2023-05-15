import argparse


def main(args: argparse.Namespace) -> None:
    # Generate the contents of the job file.
    sh_file = f"""#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name={args.model}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm/out/{args.model}_%A.txt

module purge
module load 2021
module load Anaconda3/2021.05

source activate CoT-XAI-HateSpeechDetection

cd $HOME/CoT-XAI-HateSpeechDetection
srun python3 src/eval.py --model {args.model} --show_preds 10 --prompt_path {args.prompt_path}

echo \"Job finished fully.\"
"""
    print(sh_file)

    # Create the job file.
    filename = f"slurm/job/{args.model}.sh"
    with open(filename, "w") as f:
        f.write(sh_file)
    print(
        "\n" + "-" * 80 + "\n" + f"A job file has been created in `{filename}`.\n"
        "Run the following command to submit the job:\n"
        f"$ sbatch {filename}"
    )


if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser()

    # Required parameters.
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # For up-to-date model names, see https://huggingface.co/MBZUAI.
        choices=[
            "LaMini-Cerebras-111M",
            "LaMini-Cerebras-256M",
            "LaMini-Cerebras-590M",
            "LaMini-Cerebras-1.3B",
            "LaMini-GPT-124M",
            "LaMini-GPT-774M",
            "LaMini-GPT-1.5B",
            "LaMini-Neo-125M",
            "LaMini-Neo-1.3B",
            "LaMini-Flan-T5-77M",
            "LaMini-Flan-T5-248M",
            "LaMini-Flan-T5-783M",
            "LaMini-T5-61M",
            "LaMini-T5-223M",
            "LaMini-T5-738M",
        ],
        help="The name of the model to evaluate.",
    )

    parser.add_argument(
        "--prompt_path", type=str, help="Path to the prompt to use for evaluation."
    )

    # Parse the arguments.
    args = parser.parse_args()
    main(args)
