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
    model_choices = [
        "MBZUAI/LaMini-Cerebras-111M",
        "MBZUAI/LaMini-Cerebras-256M",
        "MBZUAI/LaMini-Cerebras-590M",
        "MBZUAI/LaMini-Cerebras-1.3B",
        "MBZUAI/LaMini-GPT-124M",
        "MBZUAI/LaMini-GPT-774M",
        "MBZUAI/LaMini-GPT-1.5B",
        "MBZUAI/LaMini-Neo-125M",
        "MBZUAI/LaMini-Neo-1.3B",
        "MBZUAI/LaMini-Flan-T5-77M",
        "MBZUAI/LaMini-Flan-T5-248M",
        "MBZUAI/LaMini-Flan-T5-783M",
        "MBZUAI/LaMini-T5-61M",
        "MBZUAI/LaMini-T5-223M",
        "MBZUAI/LaMini-T5-738M",
    ]
    print("-" * 80)
    print()
    for model_choice in model_choices:
        print(
            "python3 slurm/job/generate_eval_job.py "
            f"--model {model_choice} "
            f"--prompt_name {args.prompt_name}"
        )
    print()
    print("-" * 80)
    print()
    for model_choice in model_choices:
        job_filename = (
            f"slurm/job/{safeguard_filename(model_choice)}_"
            f"{safeguard_filename(args.prompt_name)}.sh"
        )
        print(f"sbatch {job_filename}")
    print("squeue -u $USER")
    print()
    print("-" * 80)


if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments.
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
