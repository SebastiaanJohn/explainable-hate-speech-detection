"""Evaluation script for the social bias model."""
import argparse
import logging
import re

import torch
from dataset.get_dataset import get_social_bias_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import pipeline


def get_prompt(text: str, prompt: str = "Is this post hateful?", prefix: bool = False) -> str:
    """Appends the text to the baseline prompt.

    Args:
        text (str): The text to use as the prompt.
        prompt (str, optional): The baseline prompt. Defaults to "Is this post hateful?".
        prefix (bool, optional): Whether to prepend or append the text to the prompt.

    Returns:
        str: The prompt.
    """
    if prefix:
        return f"{prompt}\n\n{text}"
    return f"{text}\n\n{prompt}"


def dataloader(dataset):
    """Returns a generator that yields the prompt for each example in the dataset."""
    for example in tqdm(dataset):
        yield get_prompt(example["text"])


def classify(response: str) -> int | None:
    """Extracts the label from the response (yes/no)."""
    response = response.lower()
    if re.search(r'\byes\b', response):
        return 1
    elif re.search(r'\bno\b', response):
        return 0
    else:
        return None

def evaluate(model: str, dataset, device: torch.device, args: argparse.Namespace):
    """Evaluate the model on the dataset.

    Args:
        model (str): The model checkpoint to use.
        dataset (Dataset): The dataset to evaluate on.
        device (torch.device): The device to use.
        args (argparse.Namespace): The arguments to the script.

    Returns:
        dict: The accuracy, F1 score, precision, and recall of the model on the dataset.
    """
    checkpoint = f"MBZUAI/{model}"
    pipe = pipeline('text2text-generation', model=checkpoint, device=device)

    true_labels = []
    predicted_labels = []

    for idx, response in enumerate(pipe(dataloader(dataset), return_tensors=False)):
        prediction = classify(response[0]["generated_text"])
        if prediction is not None:
            true_labels.append(1 if dataset[idx]["is_offensive"] == 'yes' else 0)
            predicted_labels.append(prediction)

    accuracy = sum([1 for t, p in zip(true_labels, predicted_labels) if t == p]) / len(true_labels)
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main(args) -> None:
    """Main function for the evaluation script."""
    logging.info(f"Args: {args}")
    logging.info(f"Starting evaluation of {args.model} on {args.split} split.")

    # Set the random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the dataset
    dataset = get_social_bias_dataset(split=args.split, subset=args.subset)

    # Evaluate the model
    logging.info(f"Evaluating {args.model}...")
    metrics = evaluate(args.model, dataset, device, args)

    # Log the results
    logging.info(f"Results for {args.model}:")
    logging.info(f"Accuracy: {metrics['accuracy']:0.3f}")
    logging.info(f"F1: {metrics['f1']:0.3f}")
    logging.info(f"Precision: {metrics['precision']:0.3f}")
    logging.info(f"Recall: {metrics['recall']:0.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("model", type=str, help="The name of the model to evaluate.")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the data loader.")
    parser.add_argument("--subset", type=int, default=None, help="Number of examples to use from the dataset.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    main(args)
