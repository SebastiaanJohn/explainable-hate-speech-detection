"""This script is used to download and process the dataset."""

import logging

from datasets import Dataset, load_dataset


def get_social_bias_dataset(split: str = "train", subset: int | None = None) -> Dataset:
    """Returns the social bias dataset.

    Args:
        split (str): The split of the dataset to return. Must be one of 'train',
        'validation', or 'test'. Defaults to 'train'.
        subset (int, optional): The number of examples to return. Defaults to None.

    Returns:
        Dataset: The social bias dataset.
    """
    if split not in {"train", "validation", "test"}:
        error = f"Invalid split: {split}. Must be one of 'train', 'validation', or 'test'."
        raise ValueError(error)

    logging.info(f"Loading Social Bias dataset split: {split}...")
    if subset is not None:
        logging.info(f"Only using {subset} examples.")
        split = f"{split}[:{subset}]"
    dataset = load_dataset("social_bias_frames", split=split)

    # We want to filter out '' from the dataset (2017 examples).
    logging.info("Filtering out examples with empty labels...")
    dataset = dataset.filter(lambda example: example["offensiveYN"] != '')

    # Create a mapping from the id to the label. We map 0.5 to "yes" as well.
    id_to_label = {
        "0.0": "no",
        "0.5": "yes",
        "1.0": "yes",
    }

    # We only want to keep the text "post", and "offensiveYN", where 0.0 is
    # non-offensive, 0.5 is maybe offensive, and 1.0 is offensive.
    logging.info("Mapping labels to 'yes' and 'no'...")
    dataset = dataset.map(
        lambda example: {
            "text": example["post"],
            "is_offensive": id_to_label[example["offensiveYN"]],
        },
    )

    # We only want to keep the text and label and remove the rest.
    dataset = dataset.remove_columns(
        ['whoTarget',
         'intentYN',
         'sexYN',
         'sexReason',
         'offensiveYN',
         'annotatorGender',
         'annotatorMinority',
         'sexPhrase',
         'speakerMinorityYN',
         'WorkerId',
         'HITId',
         'annotatorPolitics',
         'annotatorRace',
         'annotatorAge',
         'post',
         'targetMinority',
         'targetCategory',
         'targetStereotype',
         'dataSource']
    )

    # Count how many examples in each class.
    count_yes = dataset.filter(lambda example: example["is_offensive"] == "yes").num_rows
    count_no = dataset.filter(lambda example: example["is_offensive"] == "no").num_rows
    logging.info(f"Number of examples with labels - 'yes': {count_yes}, 'no': {count_no}")

    logging.info("Done loading social bias dataset.")

    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = get_social_bias_dataset()
    print(dataset[0])
