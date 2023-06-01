import argparse


def main(args: argparse.Namespace) -> None:
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
            f"--prompt_name {args.prompt_name}"
        )
    print()
    print("-" * 80)
    print()
    for model_choice in model_choices:
        print(f"sbatch slurm/job/{model_choice}_{args.prompt_name}.sh")
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
