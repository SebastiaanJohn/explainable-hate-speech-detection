"""Evaluation script for the social bias model."""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from tabulate import tabulate


# isort: off
from utils import (
    DIR_CACHES,
    DIR_IMAGES,
    DIR_PROMPTS,
    extract_prompt,
    safeguard_filename,
)
from data.get_dataset import get_social_bias_dataset
from generation.generate import generate_predictions

# isort: on


def confusion_matrix(
    labels_true: list[int], labels_pred: list[int]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute the confusion matrix.

    Args:
        labels_true (list[int]): The true labels.
        labels_pred (list[int]): The predicted labels.

    Returns:
        Tuple containing:
            np.ndarray: The confusion matrix, where the rows represent the true
                labels and the columns represent the predicted labels.
            list[str]: The labels for the rows.
            list[str]: The labels for the columns.
    """
    row_labels = sorted(set(labels_true))
    col_labels = sorted(set(labels_pred))

    # Map the row/column indices to labels.
    rowlabel2idx = {label: idx for idx, label in enumerate(row_labels)}
    collabel2idx = {label: idx for idx, label in enumerate(col_labels)}

    # Initialize the confusion matrix.
    cm = np.zeros((len(row_labels), len(col_labels)), dtype=int)

    # Fill the confusion matrix.
    for label_true, label_pred in zip(labels_true, labels_pred):
        cm[rowlabel2idx[label_true], collabel2idx[label_pred]] += 1

    return cm, row_labels, col_labels


def evaluate(
    model: str, labels_true: list[str], labels_pred: list[str]
) -> dict:
    """Evaluate the generated predictions on the dataset.

    Args:
        model (str): The name of the model.
        labels_true (list[str]): The ground truth labels.
        labels_pred (list[str]): The predicted labels.

    Returns:
        Dict containing:
            confusion_matrix (tuple[np.ndarray, list[str], list[str]]): The
                confusion matrix, where the rows represent the true labels and
                the columns represent the predicted labels.
            balanced_accuracy (float): The balanced accuracy of the model.
                Calculated as the average of recall obtained on each class.
    """
    logging.info(f"Evaluating {model} on {len(labels_true)} examples...")
    return {
        "confusion_matrix": confusion_matrix(labels_true, labels_pred),
        "balanced_accuracy": balanced_accuracy_score(labels_true, labels_pred),
    }


def show_metrics(metrics: dict) -> None:
    """Show the metrics produced by the evaluation function.

    Args:
        metrics (dict): The metrics produced by the evaluation function.
    """
    # Show the confusion matrix.
    cm, row_labels, col_labels = metrics["confusion_matrix"]
    cm = np.vstack((col_labels, cm))
    row_labels.insert(0, r"True " + "\u2193" + " Predicted " + "\u2192")
    cm_table = tabulate(cm, showindex=row_labels, tablefmt="fancy_grid")
    logging.info(f"Confusion matrix:\n{cm_table}")

    # Show the quantitative metrics.
    logging.info(f"Balanced accuracy: {metrics['balanced_accuracy']:0.3f}")


def show_confidence_histogram(
    labels_true: list[str],
    probs_positive: list[float],
    model: str,
    prompt_name: str,
    dir_images: str,
) -> None:
    """Show a histogram of the model's confidence vs the actual label.

    Args:
        labels_true (list[str]): The ground truth labels.
        probs_positive (list[float]): For every sample, the model's
            confidence that it should be classified as positive.
        model (str): The name of the model.
        prompt_name (str): The name of the prompt.
        dir_images (str): Path to the directory where the images are saved.
    """
    # Create the histograms.
    probs_positive = np.array(probs_positive) * 100
    labels_true = np.array(labels_true)
    plt.hist(
        probs_positive[labels_true == "yes"],
        bins=np.linspace(0, 100, 21),
        color="green",
        alpha=0.5,
        label="Ground truth is 'yes'",
    )
    plt.hist(
        probs_positive[labels_true == "no"],
        bins=np.linspace(0, 100, 21),
        color="red",
        alpha=0.5,
        label="Ground truth is 'no'",
    )
    plt.title(
        "Model's confidence vs actual label\n"
        f"Model: {model}\n"
        f"Prompt: {prompt_name}"
    )
    plt.xlabel("Confidence of 'yes' (%)")
    plt.ylabel("Number of predictions")
    plt.xlim(0, 100)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()

    # Save the plot.
    path_plot = safeguard_filename(f"{model}_{prompt_name}.png")
    path_plot = os.path.join(dir_images, path_plot)
    plt.savefig(path_plot)
    logging.info(f"Saved confidence histogram to {path_plot}.")
    plt.clf()


def main(args: argparse.Namespace) -> None:
    """Main function for the evaluation script.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # Set the random seed.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create the directories for the caches, images and prompts if they don't
    # already exist.
    os.makedirs(args.dir_caches, exist_ok=True)
    os.makedirs(args.dir_images, exist_ok=True)
    os.makedirs(args.dir_prompts, exist_ok=True)

    # Extract the prompt.
    prompt_template = extract_prompt(args.dir_prompts, args.prompt_name)

    # Load the dataset.
    dataset = get_social_bias_dataset(args.split, num_workers=args.num_workers)

    # Generate the predictions.
    predictions = generate_predictions(
        dataset,
        prompt_template,
        args.model,
        args.split,
        args.max_length,
        args.save_every,
        args.preds_to_gen,
        args.preds_to_show,
        args.labels_positive,
        args.labels_negative,
        args.use_mps,
        args.dir_caches,
    )
    probs_positive = [p["prob_positive"] for p in predictions]

    # Extract the labels.
    labels_true = dataset["offensiveYN"][: len(probs_positive)]
    labels_pred = ["yes" if c > 0.5 else "no" for c in probs_positive]

    # Evaluate the model.
    metrics = evaluate(args.model, labels_true, labels_pred)

    # Show the metrics.
    show_metrics(metrics)

    # Show the confidence histogram.
    show_confidence_histogram(
        labels_true,
        probs_positive,
        args.model,
        args.prompt_name,
        args.dir_images,
    )


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

    # Optional arguments.
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="The logging level to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to use in NumPy, PyTorch, and CuDNN.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for data loading. If set to 0, "
        "multiprocessing will not be used.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="The split of the dataset to evaluate on.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum amount of tokens the model can generate for its "
        "explanation.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Number of predictions to generate before caching them.",
    )
    parser.add_argument(
        "--preds_to_gen",
        type=int,
        default=0,
        help="Minimum number of predictions to generate. If the cache file "
        "already contains some of these predictions, the model will continue "
        "generating predictions until this number is reached. If 0, all "
        "predictions will be generated.",
    )
    parser.add_argument(
        "--preds_to_show",
        type=int,
        default=10,
        help="Number of generated predictions that should be logged.",
    )
    parser.add_argument(
        "--labels_positive",
        type=str,
        nargs="+",
        default=["Yes", "yes"],
        help="Labels that are considered positive.",
    )
    parser.add_argument(
        "--labels_negative",
        type=str,
        nargs="+",
        default=["No", "no"],
        help="Labels that are considered negative.",
    )
    parser.add_argument(
        "--use_mps",
        action="store_true",
        help="If True, the model is evaluated on an Apple GPU.",
    )
    parser.add_argument(
        "--dir_caches",
        type=str,
        default=DIR_CACHES,
        help="Path to the directory where the caches are stored.",
    )
    parser.add_argument(
        "--dir_images",
        type=str,
        default=DIR_IMAGES,
        help="Path to the directory where the images are saved.",
    )
    parser.add_argument(
        "--dir_prompts",
        type=str,
        default=DIR_PROMPTS,
        help="Path to the directory where the prompts are saved.",
    )

    # Parse the arguments.
    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(
        level=args.logging_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Print command line arguments.
    logging.info(f"{args=}")

    main(args)
