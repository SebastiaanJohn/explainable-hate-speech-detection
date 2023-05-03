"""This script is used to download and process the dataset."""

import argparse
import logging

from datasets import Dataset, load_dataset


def get_social_bias_dataset(split: str, num_workers: int = 0) -> Dataset:
    """
    Returns the Social Bias Frames dataset.

    Args:
        split (str): The split of the dataset to load. Must be one of "train",
            "validation", or "test".
        num_workers (int, optional): Number of workers to use for data loading.
            If set to 0, no multiprocessing will be used. Defaults to 0.

    Returns:
        Dataset: The social bias dataset, with the following columns:
            post (str): The text of the post.
            offensiveYN (str): The label of the post. Either "yes" or "no".
    """
    if split not in ("train", "validation", "test"):
        raise ValueError(
            f"Invalid split: '{split}'. Must be one of 'train', "
            "'validation', or 'test'."
        )

    # Load the dataset.
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

    # We want to filter out "" from the dataset (2017 examples).
    logging.info("Filtering out examples with empty labels...")
    dataset = dataset.filter(
        lambda example: example["offensiveYN"] != "",
        num_proc=None if num_workers == 0 else num_workers,
    )

    # Create a mapping from the id to the label. We consider 0.0 to be
    # non-offensive, and 0.5 and 1.0 to be offensive.
    logging.info("Mapping labels to 'yes' and 'no'...")
    id_to_label = {"0.0": "no", "0.5": "yes", "1.0": "yes"}
    dataset = dataset.map(
        lambda example: {"offensiveYN": id_to_label[example["offensiveYN"]]},
        num_proc=None if num_workers == 0 else num_workers,
    )

    # Count how many examples are in each class.
    for label in sorted(set(id_to_label.values())):
        count_label = dataset.filter(
            lambda example: example["offensiveYN"] == label
        ).num_rows
        logging.info(f"Number of examples with label '{label}': {count_label}")
    logging.info(f"Total number of examples: {dataset.num_rows}")

    return dataset


def main(args: argparse.Namespace):
    dataset = get_social_bias_dataset(args.split, num_workers=args.num_workers)
    logging.info(f"{next(iter(dataset))=}")


if __name__ == "__main__":
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create the argument parser.
    parser = argparse.ArgumentParser()

    # Optional parameters.
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Split to evaluate on. Defaults to 'test'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for data loading. If set to 0, "
        "no multiprocessing will be used. Defaults to 8.",
    )

    # Parse the arguments.
    args = parser.parse_args()
    main(args)
