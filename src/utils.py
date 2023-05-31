"""This file contains general utility functions for the project."""

import logging
import os


# Define global variables.
DIR_CACHES = "./data/caches"
DIR_IMAGES = "./data/images"
DIR_PROMPTS = "./data/prompts"
VALID_FILENAME_CHARS = (
    "-=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
pred_type = dict[str, str | float]


def extract_prompt(dir_prompts: str, prompt_name: str) -> str:
    """Extracts a prompt template from a file.

    Args:
        dir_prompts (str): Path to the directory where the prompts are saved.
        prompt_name (str): Name of the prompt.

    Returns:
        str: The extracted prompt template.
    """
    with open(os.path.join(dir_prompts, f"{prompt_name}.txt")) as f:
        prompt_template = f.read()

    if "{post}" not in prompt_template:
        raise ValueError(
            "The prompt must contain the string '{post}' to indicate where "
            "the post should be inserted."
        )
    if "{answer}" not in prompt_template:
        raise ValueError(
            "The prompt must contain at least the string '{answer}' to "
            "indicate where the model should be asked to predict the label."
        )

    logging.info(
        "\n>>> ".join(["Prompt template:"] + prompt_template.split("\n"))
    )
    return prompt_template


def safeguard_filename(filename: str) -> str:
    """Converts a filename to a safe filename by removing special characters.

    Args:
        filename (str): The filename to convert.

    Returns:
        str: The converted filename.
    """
    return "".join(c if c in VALID_FILENAME_CHARS else "-" for c in filename)


def log_prediction(
    example_idx: int,
    post: str,
    label_true: str,
    prob_positive: float,
    explanation: str = "",
) -> None:
    """Show a single prediction.

    Args:
        example_idx (int): The index of the example.
        post (str): The post.
        label_true (str): The ground truth label.
        prob_positive (float): The probability of the positive class.
        explanation (str, optional): The explanation for the prediction.
            Defaults to "".
    """
    logging.info(f" Example {example_idx} ".center(65, "-"))
    logging.info("\n>>> ".join(["Input post:"] + post.split("\n")))
    logging.info(
        "\n>>> ".join(["Model explanation:"] + explanation.split("\n"))
    )
    logging.info(f"Ground truth label: {label_true}")
    logging.info(f"Predicted probability of positive class: {prob_positive}")
    logging.info("-" * 65)
