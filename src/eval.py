"""Evaluation script for the social bias model."""
import argparse
import logging
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import torch
from dataset.get_dataset import get_social_bias_dataset
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import pipeline


def safeguard_filename(filename: str) -> str:
    """
    Converts a filename to a safe filename by removing all special characters.

    Args:
        filename (str): The filename to convert.

    Returns:
        str: The converted filename.
    """
    valid_chars = (
        "-_.=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    return "".join(c if c in valid_chars else "-" for c in filename)


def determine_preds_cache_location(
    model: str,
    split: str,
    prompt: str,
    preds_cache_dir: str = "./data/preds_cache",
) -> str:
    """
    Determines the cache location where the predictions are/should be stored.

    If a cache file already exists for the given model, split, and prompt,
    that cache file is returned. Otherwise, the location where the predictions
    should be stored is returned. You can check whether the returned cache file
    location already exists by using `os.path.exists(cache_location)`.

    Args:
        model (str): The model checkpoint to use.
        split (str): The split of the dataset to use. Must be one of "train",
            "validation", or "test".
        prompt (str, optional): The prompt to be prepended to each post.
        preds_cache_dir (str, optional): The directory where the cache files
            are stored. Defaults to "./data/preds_cache".

    Returns:
        str: The location of the cache file.
    """
    # Create the cache directory if it doesn't exist yet.]
    os.makedirs(preds_cache_dir, exist_ok=True)

    # Look for all files in the cache directory that match the model, split,
    # and prompt.
    safe_model = safeguard_filename(model)
    safe_split = safeguard_filename(split)
    # `match.group(0)` is the cache file name.
    # `match.group(1)` is prompt identifier.
    # `match.group(2)` is the amount of predictions that have been generated.
    matches_all = [
        re.fullmatch(
            f"model={safe_model}_split={safe_split}_"
            r"prompt-id=(\d+)_progress=(\d+)\.pkl",
            filename,
        )
        for filename in os.listdir(preds_cache_dir)
    ]

    # First group the matches by prompt so that we can later search for the
    # file that contains the current prompt.
    versions = defaultdict(list)
    for match in matches_all:
        if match is not None:
            versions[int(match.group(1))].append(match)
    for matches in versions.values():
        matches.sort(key=lambda match: int(match.group(2)), reverse=True)

    # Now search for the cache file that contains the current prompt.
    # If the prompt is found, we don't exit immediately because we this is a
    # good oppertunity to check the integrety of the cache files (more
    # specifically, we want to check whether all cache files with the same
    # prompt identifier contain the same prompt).
    cache_location = None
    for matches in versions.values():
        prev_prompt = None
        prev_location = None
        for match in matches:
            curr_location = os.path.join(preds_cache_dir, match.group(0))
            with open(curr_location, "rb") as f:
                content = pickle.load(f)
                if prev_prompt is None:
                    if content["prompt"] == prompt:
                        cache_location = curr_location
                    prev_prompt = content["prompt"]
                elif content["prompt"] != prev_prompt:
                    raise ValueError(
                        "All cache files with the same prompt identifier "
                        "should contain the same prompt. This error should "
                        "never occur, unless a cache was manually modified.\n"
                        f"Prompt 1: {prev_prompt} in {prev_location}.\n"
                        f"Prompt 2: {content['prompt']} in {curr_location}."
                    )

    # If a cache file was found, return it.
    if cache_location is not None:
        return cache_location

    # Choose a prompt identifier that isn't already taken.
    prompt_id = 0
    while prompt_id in versions:
        prompt_id += 1

    # Return the new cache file name.
    return os.path.join(
        preds_cache_dir,
        f"model={safe_model}_split={safe_split}_"
        f"prompt-id={prompt_id}_progress=0.pkl",
    )


def update_preds_cache(
    cache_location: str, prompt: str, predictions: list[str]
) -> str:
    """
    Updates the predictions cache file with the given predictions.

    Args:
        cache_location (str): The location of the cache file.
        predictions (list[str]): The predictions to add to the cache file.
        prompt (str, optional): The prompt to be prepended to each post.

    Returns:
        str: The new location of the cache file.
    """
    # Update the predictions cache file.
    new_cache_location = re.sub(
        r"progress=\d+", f"progress={len(predictions)}", cache_location
    )
    with open(new_cache_location, "wb") as f:
        pickle.dump({"prompt": prompt, "predictions": predictions}, f)

    # Delete the old cache file.
    if new_cache_location != cache_location and os.path.exists(cache_location):
        os.remove(cache_location)

    return new_cache_location


def generate_predictions(
    model: str,
    split: str,
    prompt: str,
    dataset: Dataset,
    minimum_preds: int = 0,
    preds_cache_dir: str = "./data/preds_cache",
    save_every: int = 1000,
) -> list[str]:
    """
    Generates predictions for the dataset using the model.

    The predictions are retrieved from a cache file if it already exists.
    This cache file is a pickle file that contains a dictionary with:
        - "prompt": The prompt used to generate the predictions.
        - "predictions": A list of prediction strings.
    A pickle file is used in favour of a "more readable" text file because
    both the prompt and the predictions can contain newlines or even empty
    lines, which would make it difficult to parse a text file.

    Args:
        model (str): The model checkpoint to use.
        split (str): The split of the dataset to generate predictions for. Must
            be one of "train", "validation", or "test".
        prompt (str): The prompt to be prepended to each post.
        dataset (Dataset): The dataset to evaluate on.
        minimum_preds (int, optional): The minimum amount of predictions that
            should be generated. If the cache file already contains some of
            these predictions, the model will continue generating predictions
            until the minimum amount is reached. If 0, all predictions will be
            generated. Defaults to 0.
        preds_cache_dir (str, optional): The directory where the cache files
            are stored. Defaults to "./data/preds_cache".
        save_every (int, optional): The amount of predictions that should be
            generated before saving them to the cache file. Defaults to 1000.

    Returns:
        list[str]: A list of predictions.
    """
    # Determine the cache location.
    cache_location = determine_preds_cache_location(
        model, split, prompt, preds_cache_dir=preds_cache_dir
    )

    # If the cache file exists:
    # - If it contains enough predictions, return the them immediately.
    # - Otherwise, load the predictions from the cache file and continue.
    # If the cache file doesn't exist, create it.
    if os.path.isfile(cache_location):
        logging.info(f"Using cached predictions from {cache_location}.")
        with open(cache_location, "rb") as f:
            content = pickle.load(f)
            if len(content["predictions"]) >= minimum_preds:
                return content["predictions"]
            else:
                predictions = content["predictions"]
    else:
        logging.info("Generating predictions from scratch.")
        predictions = []

    # Generate the (missing) predictions.
    # To prevent crashes or other errors from causing the loss of all
    # computed predictions, we save the predictions to the cache file after
    # every `save_every` predictions.
    pipe = pipeline("text2text-generation", model=f"MBZUAI/{model}")
    for response in pipe(
        iter(
            tqdm(
                (
                    prompt + "\n" + post
                    for post in dataset[len(predictions) :]["post"]
                ),
                initial=len(predictions),
                total=len(dataset),
            )
        )
    ):
        predictions.append(response[0]["generated_text"])
        if len(predictions) % save_every == 0:
            cache_location = update_preds_cache(
                cache_location, prompt, predictions
            )

    # Update the cache file with the final predictions.
    update_preds_cache(cache_location, prompt, predictions)

    return predictions


def classify_contains_yn(prediction: str) -> int | None:
    """
    Extract the label from the response (yes/no).

    Args:
        response (str): The response from the model.

    Returns:
        int | None: The label (1 for yes, 0 for no) or None if the response
            doesn't contain a label.
    """
    prediction = prediction.lower()
    if re.search(r"\byes\b", prediction):
        return 1
    elif re.search(r"\bno\b", prediction):
        return 0
    else:
        return None


def confusion_matrix(
    labels_true: list[int], labels_pred: list[int]
) -> tuple[np.ndarray, list[str]]:
    """
    Compute the confusion matrix.

    Args:
        labels_true (list[int]): The true labels.
        labels_pred (list[int]): The predicted labels.

    Returns:
        Tuple containing:
            np.ndarray: The confusion matrix, where the rows represent the true
                labels and the columns represent the predicted labels.
            list[str]: A list mapping the row/column indices to their
                corresponding label names.
    """
    # Extract all valid labels.
    labels = set(labels_true + labels_pred)

    # Map the row/column indices to labels.
    idx2label = sorted(labels)
    label2idx = {label: idx for idx, label in enumerate(idx2label)}

    # Initialize the confusion matrix.
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    # Fill the confusion matrix.
    for label_true, label_pred in zip(labels_true, labels_pred):
        cm[label2idx[label_true], label2idx[label_pred]] += 1

    return cm, idx2label


def evaluate(
    model: str,
    split: str,
    prompt: str,
    dataset: Dataset,
    minimum_preds: int = 0,
    preds_cache_dir: str = "./data/preds_cache",
    save_every: int = 1000,
) -> dict[str, float]:
    """
    Evaluate the model on the dataset.

    Args:
        model (str): The model checkpoint to use.
        split (str): The split of the dataset to use. Must be one of "train",
            "validation", or "test".
        prompt (str): The prompt to be prepended to each post.
        dataset (Dataset): The dataset to evaluate on.
        minimum_preds (int, optional): The minimum amount of predictions that
            should be generated. If the cache file already contains some of
            these predictions, the model will continue generating predictions
            until the minimum amount is reached. If 0, all predictions will be
            generated. Defaults to 0.
        preds_cache_dir (str, optional): The directory where the cache files
            are stored. Defaults to "./data/preds_cache".
        save_every (int, optional): The amount of predictions that should be
            generated before saving them to the cache file. Defaults to 1000.

    Returns:
        dict: The accuracy, F1 score, precision, and recall of the model on
            the dataset.
    """
    # First get the model's predictions.
    predictions = generate_predictions(
        model,
        split,
        prompt,
        dataset,
        minimum_preds=minimum_preds,
        preds_cache_dir=preds_cache_dir,
        save_every=save_every,
    )

    # Then extract predicted and ground truth labels.
    labels_pred = [
        classify_contains_yn(prediction) for prediction in predictions
    ]
    labels_true = [example["offensiveYN"] == "yes" for example in dataset]

    # Finally evaluate the predictions.
    cm, idx2label = confusion_matrix(labels_true, labels_pred)
    accuracy = accuracy_score(labels_true, labels_pred)
    f1 = f1_score(labels_true, labels_pred)
    precision = precision_score(labels_true, labels_pred)
    recall = recall_score(labels_true, labels_pred)

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main(args: argparse.Namespace):
    """
    Main function for the evaluation script.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    logging.info(f"Args: {args}")

    # Set the random seed.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the dataset.
    dataset = get_social_bias_dataset(
        args.split,
        minimum_preds=args.minimum_preds,
        num_workers=args.num_workers,
    )
    args.minimum_preds = len(dataset)

    # Extract the prompt.
    with open(args.prompt_path, "r") as f:
        prompt = f.read()

    # Evaluate the model.
    metrics = evaluate(
        args.model,
        args.split,
        prompt,
        dataset,
        minimum_preds=args.minimum_preds,
        preds_cache_dir=args.preds_cache_dir,
        save_every=args.save_every,
    )

    # Log the results.
    # TODO Show the confusion matrix.
    logging.info(f"Results for {args.model}:")
    logging.info(f"Accuracy: {metrics['accuracy']:0.3f}")
    logging.info(f"F1: {metrics['f1']:0.3f}")
    logging.info(f"Precision: {metrics['precision']:0.3f}")
    logging.info(f"Recall: {metrics['recall']:0.3f}")


if __name__ == "__main__":
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Craete the argument parser.
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
            "LaMini-GPT-774M",
            "LaMini-GPT-124M",
            "LaMini-GPT-1.5B",
            "LaMini-Neo-125M",
            "LaMini-Neo-1.3B",
            "LaMini-Flan-T5-783M",
            "LaMini-Flan-T5-248M",
            "LaMini-Flan-T5-77M",
            "LaMini-T5-738M",
            "LaMini-T5-223M",
            "LaMini-T5-61M",
            "bactrian-x-7b-lora",
            "swiftformer-l1",
            "swiftformer-l3",
            "swiftformer-s",
            "swiftformer-xs",
        ],
        help="The name of the model to evaluate.",
    )

    # Optional parameters.
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Defaults to 42."
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Split to evaluate on. Defaults to 'test'.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./prompts/default.txt",
        help="Path to the prompt to use for evaluation. Defaults to "
        "'./prompts/default.txt'.",
    )
    parser.add_argument(
        "--minimum_preds",
        type=int,
        default=0,
        help="The minimum amount of predictions that should be generated. If "
        "the cache file already contains some of these predictions, the model "
        "will continue generating predictions until the minimum amount is "
        "reached. If 0, all predictions will be generated. Defaults to 0.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for data loading. If set to 0, "
        "no multiprocessing will be used. Defaults to 8.",
    )
    parser.add_argument(
        "--preds_cache_dir",
        type=str,
        default="./data/preds_cache",
        help="Path to the directory where the predictions cache should be "
        "stored. Defaults to './data/preds_cache'.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Cache the predictions every n predictions. Defaults to 1000.",
    )

    # Parse the arguments.
    args = parser.parse_args()
    main(args)
