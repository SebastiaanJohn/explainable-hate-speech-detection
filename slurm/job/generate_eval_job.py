import argparse


VALID_FILENAME_CHARS = (
    "-=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def safeguard_filename(filename: str) -> str:
    """Converts a filename to a safe filename by removing special characters.

    Args:
        filename (str): The filename to convert.

    Returns:
        str: The converted filename.
    """
    return "".join(c if c in VALID_FILENAME_CHARS else "-" for c in filename)


def main(args: argparse.Namespace) -> None:
    out_filename = (
        f"slurm/out/{safeguard_filename(args.model)}_"
        f"{safeguard_filename(args.prompt_name)}_%A.txt"
    )
    job_name = (
        f"{safeguard_filename(args.model)}_"
        f"{safeguard_filename(args.prompt_name)}"
    )

    # Create the job file.
    sh_file = rf"""#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name={job_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --mem=32000M
#SBATCH --output={out_filename}

module purge
module load 2021
module load Anaconda3/2021.05

source activate CoT-XAI-HateSpeechDetection

cd $HOME/CoT-XAI-HateSpeechDetection
srun python3 src/eval.py \
    --model {args.model} \
    --prompt_name {args.prompt_name}

echo "Job finished fully."
"""
    print(sh_file)

    # Create the job file.
    job_filename = (
        f"slurm/job/{safeguard_filename(args.model)}_"
        f"{safeguard_filename(args.prompt_name)}.sh"
    )
    with open(job_filename, "w") as f:
        f.write(sh_file)
    print(
        "\n"
        + "-" * 80
        + "\n"
        + f"A job file has been created in `{job_filename}`.\n"
        + "Run the following command to submit the job:\n"
        + f"$ sbatch {job_filename}"
    )


if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required parameters.
    # Required arguments.
    # Common values (I put them here so we can easily copy-paste them):
    # MBZUAI/LaMini-T5-61M
    # MBZUAI/LaMini-T5-223M
    # MBZUAI/LaMini-T5-738M
    # MBZUAI/LaMini-Flan-T5-77M
    # MBZUAI/LaMini-Flan-T5-248M
    # MBZUAI/LaMini-Flan-T5-783M
    # MBZUAI/LaMini-Cerebras-111M
    # MBZUAI/LaMini-Cerebras-256M
    # MBZUAI/LaMini-Cerebras-590M
    # MBZUAI/LaMini-Cerebras-1.3B
    # MBZUAI/LaMini-Neo-125M
    # MBZUAI/LaMini-Neo-1.3B
    # MBZUAI/LaMini-GPT-124M
    # MBZUAI/LaMini-GPT-774M
    # MBZUAI/LaMini-GPT-1.5B
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="The name of the model to evaluate.",
    )
    # Common values (I put them here so we can easily copy-paste them):
    # default_0shot_no_cot
    # default_0shot_cot_expl_first
    # default_4shot_cot_ans_first
    # default_4shot_cot_expl_first
    parser.add_argument(
        "--prompt_name",
        required=True,
        type=str,
        help="Name of the prompt. The path to the prompt file is assumed to "
        "be f'{args.dir_prompts}/{args.prompt_name}.txt'.",
    )

    # Parse the arguments.
    args = parser.parse_args()
    main(args)
