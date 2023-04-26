"""This script is used to download and process the dataset."""

from datasets import Dataset, load_dataset


def get_social_bias_dataset(split: str = "train") -> Dataset:
    """Returns the social bias dataset.

    Args:
        split (str): The split of the dataset to return. Must be one of 'train',
        'validation', or 'test'. Defaults to 'train'.

    Returns:
        Dataset: The social bias dataset.
    """
    if split not in {"train", "validation", "test"}:
        error = f"Invalid split: {split}. Must be one of 'train', 'validation', or 'test'."
        raise ValueError(error)

    dataset = load_dataset("social_bias_frames", split=split)

    # We want to filter out '' from the dataset (2017 examples).
    dataset = dataset.filter(lambda example: example["offensiveYN"] != '')

    # Create a mapping from the id to the label.
    id_to_label = {
        "0.0": "no",
        "0.5": "maybe",
        "1.0": "yes",
    }

    # We only want to keep the text "post", and "offensiveYN", where 0.0 is
    # non-offensive, 0.5 is maybe offensive, and 1.0 is offensive.
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


    return dataset

if __name__ == "__main__":
    dataset = get_social_bias_dataset()
    print(dataset)
    print(dataset[0])
