"""This script is used to download and process the dataset."""

import argparse
import logging
from collections import defaultdict

import numpy as np
from datasets import Dataset, load_dataset
from unidecode import unidecode


def get_social_bias_dataset(
    split: str,
    num_workers: int = 0,
) -> Dataset:
    """Returns the Social Bias Frames dataset.

    Args:
        split (str): The split of the dataset to load. Must be one of "train",
            "validation", or "test".
        num_workers (int, optional): Number of workers to use for data loading.
            If set to 0, no multiprocessing will be used. Defaults to 0.

    Returns:
        Dataset: The social bias dataset, with the following columns:
            post (str): The text of the post.
            offensiveYN (str): The label of the post. Either the given positive
                or negative label.
    """
    if split not in ("train", "validation", "test"):
        raise ValueError(
            f"Invalid split: '{split}'. Must be one of 'train', "
            "'validation', or 'test'."
        )

    # Load the dataset (test split has 17501 rows).
    logging.info(f"Loading 'Social Bias Frames' {split} split...")
    dataset = load_dataset(
        "social_bias_frames",
        split=split,
        num_proc=None if num_workers == 0 else num_workers,
    )

    # We only want to keep the "post" and "offensiveYN" and remove the rest.
    # Perform this operation first to speed up future operations.
    logging.info("Filtering out unnecessary columns...")
    dataset = dataset.remove_columns(
        [
            "whoTarget",
            "intentYN",
            "sexYN",
            "sexReason",
            "annotatorGender",
            "annotatorMinority",
            "sexPhrase",
            "speakerMinorityYN",
            "WorkerId",
            "HITId",
            "annotatorPolitics",
            "annotatorRace",
            "annotatorAge",
            "targetMinority",
            "targetCategory",
            "targetStereotype",
            "dataSource",
        ]
    )

    # We want to filter out "" from the dataset (test split has 232 of these).
    logging.info("Filtering out examples with empty labels...")
    dataset = dataset.filter(
        lambda example: example["offensiveYN"] != "",
        num_proc=None if num_workers == 0 else num_workers,
    )

    # Aggregate posts (test split has 12578 of these). Huggingface's Dataset
    # object does not support aggregation, so we have to do this manually. For
    # each post, we look at its offensiveYN labels (each was given by a
    # different human annotator) and then take the mean. If the mean is >= 0.5,
    # we consider the post to be offensive. Otherwise, we consider it to be
    # non-offensive.
    logging.info("Aggregating posts...")
    labels_per_post = defaultdict(list)
    for example in dataset:
        labels_per_post[example["post"]].append(float(example["offensiveYN"]))
    dataset = Dataset.from_dict(
        {
            "post": list(labels_per_post.keys()),
            "offensiveYN": [
                "yes" if np.mean(labels) >= 0.5 else "no"
                for labels in labels_per_post.values()
            ],
        },
        features=dataset.features,
        info=dataset.info,
        split=dataset.split,
    )

    # Convert Unicode characters to ASCII (test split has 382 of these).
    logging.info("Converting Unicode characters to ASCII...")
    dataset = dataset.map(
        lambda example: {
            "post": unidecode(example["post"]),
        },
        num_proc=None if num_workers == 0 else num_workers,
    )

    # Count how many examples are in each class.
    logging.info("Counting examples per class...")
    for label in ("yes", "no"):
        count_label = dataset.filter(
            lambda example: example["offensiveYN"] == label
        ).num_rows
        logging.info(f"Number of examples with label '{label}': {count_label}")
    logging.info(f"Total number of examples: {dataset.num_rows}")

    return dataset


def main(args: argparse.Namespace):
    dataset = get_social_bias_dataset(args.split, num_workers=args.num_workers)
    logging.info(f"{dataset[:5]=}")


if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Define command line arguments.
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="The logging level to use.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Split to evaluate on.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for data loading. If set to 0, "
        "no multiprocessing will be used.",
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
