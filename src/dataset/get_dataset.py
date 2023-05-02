"""This script is used to download and process the dataset."""

import logging

from datasets import Dataset, load_dataset


def get_social_bias_dataset(
    split: str,
    minimum_preds: int = 0,
    num_workers: int = 0,
) -> Dataset:
    """
    Returns the social bias dataset.

    Args:
        split (str): The split of the dataset to use. Must be one of "train",
            "validation", or "test".
        minimum_preds (int, optional): The minimum amount of predictions that
            should be generated. If the cache file already contains some of
            these predictions, the model will continue generating predictions
            until the minimum amount is reached. If 0, all predictions will be
            generated. Defaults to 0.
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

    # Check whether the split contains at least the minimum amount of
    # predictions we want to generate. If not, we will warn the user and
    # change the minimum to the size of the split.
    if minimum_preds != 0:
        dataset = load_dataset("social_bias_frames", split=split)
        if dataset.num_rows < minimum_preds:
            logging.warning(
                f"Split '{split}' contains less than {minimum_preds} "
                f"examples. Changing minimum_preds to {dataset.num_rows}."
            )
            minimum_preds = dataset.num_rows
        logging.info(f"Only using {minimum_preds} examples.")
        split += f"[:{minimum_preds}]"

    # Load the dataset split.
    logging.info(f"Loading Social Bias dataset split: {split}...")
    dataset = load_dataset(
        "social_bias_frames",
        split=split,
        num_proc=None if num_workers == 0 else num_workers,
    )

    # We only want to keep the "post" and "offensiveYN" and remove the rest.
    # Perform this operation first to speed up future operations.
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
    count_yes = dataset.filter(
        lambda example: example["offensiveYN"] == "yes"
    ).num_rows
    count_no = dataset.filter(
        lambda example: example["offensiveYN"] == "no"
    ).num_rows
    logging.info(
        "Number of examples with labels: "
        f"'yes': {count_yes}, 'no': {count_no}"
    )

    logging.info("Done loading social bias dataset.")

    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = get_social_bias_dataset("train")
    print(f"{dataset[0]=}")
