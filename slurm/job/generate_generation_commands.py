import argparse
import os


def main(args: argparse.Namespace) -> None:
    # Generate the contents of the job file.
    prompt_name = os.path.basename(args.prompt_path)
    # Get the file name without the extension.
    prompt_name = os.path.splitext(prompt_name)[0]

    model_choices = [
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
    ]
    print("-" * 80)
    print()
    for model_choice in model_choices:
        print(
            "python3 slurm/job/generate_eval_job.py "
            f"--prompt_path {args.prompt_path} "
            f"--model {model_choice} "
            f"--max_length {args.max_length}"
        )
    print()
    print("-" * 80)
    print()
    for model_choice in model_choices:
        print(f"sbatch slurm/job/{model_choice}_{prompt_name}.sh")
    print("squeue -u $USER")
    print()
    print("-" * 80)


if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser()

    # Define command line arguments.
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Path to the prompt to use for evaluation.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
    )

    # Parse the arguments.
    args = parser.parse_args()
    main(args)
