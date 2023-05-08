"Utility functions for generating or loading model predictions."

import logging
import os
import pickle
import re
from collections import defaultdict

from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline


PROMPTS_DIR = "./prompts"
PREDS_CACHE_DIR = "./data/preds_cache"
VALID_FILENAME_CHARS = (
    r"-=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def safeguard_filename(filename: str) -> str:
    """
    Converts a filename to a safe filename by removing all special characters.

    Args:
        filename (str): The filename to convert.

    Returns:
        str: The converted filename.
    """
    return "".join(c if c in VALID_FILENAME_CHARS else "-" for c in filename)


def determine_preds_cache_location(
    model: str, split: str, prompt: str, preds_cache_dir: str = PREDS_CACHE_DIR
) -> str:
    f"""
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
            are stored. Defaults to "{PREDS_CACHE_DIR}".

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
                        f"Prompt 1: '{prev_prompt}' in {prev_location}.\n"
                        f"Prompt 2: '{content['prompt']}' in {curr_location}."
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


def extract_cache_file_info(cache_location: str) -> tuple[str, str, str]:
    """
    Extracts information from the cache file name and contents.

    Args:
        cache_location (str): The location of the cache file.

    Returns:
        Tuple containing:
            model (str): The model checkpoint used.
            split (str): The split of the dataset used.
            prompt_path (str): The path to the prompt file used.
    """
    # Extract the model, split, prompt identifier, and progress from the cache
    # file name.
    match = re.fullmatch(
        rf"model=([{VALID_FILENAME_CHARS}]+)_"
        rf"split=([{VALID_FILENAME_CHARS}]+)_"
        r"prompt-id=\d+_"
        r"progress=(\d+)\.pkl",
        os.path.basename(cache_location),
    )
    if match is None:
        raise ValueError(
            "The cache file name is not in the expected format. This error "
            "should never occur, unless a cache file name was manually "
            "modified."
        )
    model = match.group(1)
    split = match.group(2)
    progress = int(match.group(3))

    # Check whether the file name reflects what is actually in the file.
    with open(cache_location, "rb") as f:
        content = pickle.load(f)
        if model != safeguard_filename(content["model"]):
            raise ValueError(
                "The cache file name does not match the model in the cache "
                "file. This error should never occur, unless a cache file "
                "name was manually modified."
            )
        if split != safeguard_filename(content["split"]):
            raise ValueError(
                "The cache file name does not match the split in the cache "
                "file. This error should never occur, unless a cache file "
                "name was manually modified."
            )
        if progress != len(content["predictions"]):
            raise ValueError(
                "The cache file name does not match the progress in the cache "
                "file. This error should never occur, unless a cache file "
                "name was manually modified."
            )
        prompt = content["prompt"]

    # Determine the prompt path using the prompt. If the prompt is not found,
    # we warn the user and create a new prompt file.
    prompt_path = None
    for filename in os.listdir(PROMPTS_DIR):
        full_path = os.path.join(PROMPTS_DIR, filename)
        if not (os.path.isfile(full_path) and filename.endswith(".txt")):
            continue
        with open(full_path, "r") as f:
            if f.read() == prompt:
                prompt_path = full_path
                break
    if prompt_path is None:
        logging.warning(
            "The prompt used to generate the cache file was not found. "
            "This should not occur, unless a prompt file was manually "
            "modified or deleted after evaluating the model. A new prompt "
            "file will be created automatically."
        )
        times_exists = 0
        while True:
            prompt_autoname = f"autoname_{hash(prompt)}_{times_exists}.txt"
            prompt_path = os.path.join(PROMPTS_DIR, prompt_autoname)
            if not os.path.exists(prompt_path):
                break
            times_exists += 1
        with open(prompt_path, "w") as f:
            f.write(prompt)
        logging.warning(f"New prompt file location: {prompt_path}")

    return model, split, prompt_path


def update_preds_cache(
    model: str,
    split: str,
    prompt: str,
    predictions: list[str],
    cache_location: str,
) -> str:
    """
    Updates the predictions cache file with the given predictions.

    Args:
        model (str): The model checkpoint used.
        split (str): The split of the dataset used.
        prompt (str, optional): The prompt to be prepended to each post.
        predictions (list[str]): The predictions to add to the cache file.
        cache_location (str): The location of the cache file.

    Returns:
        str: The new location of the cache file.
    """
    # Update the predictions cache file.
    new_cache_location = re.sub(
        r"progress=\d+", f"progress={len(predictions)}", cache_location
    )
    with open(new_cache_location, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "split": split,
                "prompt": prompt,
                "predictions": predictions,
            },
            f,
        )

    # Delete the old cache file.
    if new_cache_location != cache_location and os.path.isfile(cache_location):
        os.remove(cache_location)

    return new_cache_location


def generate_predictions(
    model: str,
    split: str,
    prompt: str,
    dataset: Dataset,
    min_preds: int = 0,
    preds_cache_dir: str = PREDS_CACHE_DIR,
    save_every: int = 1000,
) -> list[str]:
    f"""
    Generates predictions using the model.

    WARNING: The generated list may be larger or smaller than `min_preds`,
        which respectively depends on the amount of cached predictions and the
        size of the dataset.

    Args:
        model (str): The model checkpoint to use.
        split (str): The split of the dataset to generate predictions for. Must
            be one of "train", "validation", or "test".
        prompt (str): The prompt to be prepended to each post.
        dataset (Dataset): The dataset to generate predictions for.
        min_preds (int, optional): The minimum amount of predictions that
            should be generated. If the cache file already contains some of
            these predictions, the model will continue generating predictions
            until the minimum amount is reached. If 0, all predictions will be
            generated. Defaults to 0.
        preds_cache_dir (str, optional): The directory where the cache files
            are stored. Defaults to "{PREDS_CACHE_DIR}".
        save_every (int, optional): The amount of predictions that should be
            generated before saving them to the cache file. Defaults to 1000.

    Returns:
        list[str]: The predictions generated by the model.
    """
    # Determine the cache location.
    # The predictions are retrieved from a cache file if it already exists.
    # This cache file is a pickle file that contains a dictionary with:
    #     - "prompt": The prompt used to generate the predictions.
    #     - "predictions": A list of prediction strings.
    # A pickle file is used in favour of a "more readable" text file because
    # both the prompt and the predictions can contain newlines or even empty
    # lines, which would make it difficult to parse a text file.
    logging.info("Determining cache location...")
    cache_location = determine_preds_cache_location(
        model, split, prompt, preds_cache_dir=preds_cache_dir
    )

    # Load existing predictions from the cache file.
    if os.path.isfile(cache_location):
        logging.info(f"Using cached predictions from {cache_location}.")
        with open(cache_location, "rb") as f:
            content = pickle.load(f)
            predictions = content["predictions"]
    else:
        logging.info("No cached predictions found.")
        predictions = []

    # Select a subset of the dataset to generate predictions for.
    if min_preds > dataset.num_rows:
        logging.warning(
            f"The minimum number of predictions requested ({min_preds}) is "
            f"greater than the size of the dataset ({dataset.num_rows}). "
            "It will be decreased accordingly."
        )
    select = (
        len(predictions),
        dataset.num_rows
        if min_preds == 0 or min_preds > dataset.num_rows
        else min_preds,
    )
    if select[0] >= select[1]:
        return predictions
    logging.info(f"Selecting examples {select[0]} to {select[1]}.")
    dataset = dataset.select(range(*select))

    # Generate the (missing) predictions.
    logging.info("Generating predictions...")
    if "${post}" not in prompt:
        raise ValueError(
            "The prompt must contain the string '${post}' to indicate where "
            "the post should be inserted."
        )
    for response in pipeline("text2text-generation", model=f"MBZUAI/{model}")(
        iter(
            tqdm(
                (
                    prompt.replace("${post}", example["post"])
                    for example in dataset
                ),
                initial=select[0],
                total=select[1],
            )
        )
    ):
        predictions.append(response[0]["generated_text"])

        # To prevent crashes or other errors from causing the loss of all
        # computed predictions, we save the predictions to the cache file after
        # every `save_every` predictions.
        if len(predictions) % save_every == 0:
            cache_location = update_preds_cache(
                model, split, prompt, predictions, cache_location
            )

    # Update the cache file with the final predictions.
    update_preds_cache(model, split, prompt, predictions, cache_location)

    return predictions


def load_predictions(
    model: str, split: str, prompt: str, preds_cache_dir: str = PREDS_CACHE_DIR
) -> list[str]:
    f"""
    Loads the predictions generated by the model.

    Args:
        model (str): The model checkpoint to use.
        split (str): The split of the dataset to generate predictions for. Must
            be one of "train", "validation", or "test".
        prompt (str): The prompt to be prepended to each post.
        preds_cache_dir (str, optional): The directory where the cache files
            are stored. Defaults to "{PREDS_CACHE_DIR}".

    Returns:
        list[str]: The predictions generated by the model.
    """
    # Determine the cache location.
    # The predictions are retrieved from a cache file if it already exists.
    # This cache file is a pickle file that contains a dictionary with:
    #     - "prompt": The prompt used to generate the predictions.
    #     - "predictions": A list of prediction strings.
    # A pickle file is used in favour of a "more readable" text file because
    # both the prompt and the predictions can contain newlines or even empty
    # lines, which would make it difficult to parse a text file.
    logging.info("Determining cache location...")
    cache_location = determine_preds_cache_location(
        model, split, prompt, preds_cache_dir=preds_cache_dir
    )

    # Load existing predictions from the cache file.
    if os.path.isfile(cache_location):
        logging.info(f"Using cached predictions from {cache_location}.")
        with open(cache_location, "rb") as f:
            content = pickle.load(f)
            return content["predictions"]

    raise FileNotFoundError(
        f"No cached predictions found at {cache_location}."
    )
