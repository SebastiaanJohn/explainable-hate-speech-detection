"""
This file contains functions to store and retrieve previously generated
predictions.

The predictions are stored in a cache file, which is a pickle file that
contains a dictionary with the following keys:
    prompt (str): The prompt used to generate the predictions.
    predictions (list[pred]): A list of previously generated predictions
        for each example in the dataset. Since the model can generate multiple
        predictions for each example, a list of lists is used.

A pickle file is used in favour of a "more readable" text file because both the
prompt and the predictions can contain newlines or even empty lines, which
would make it difficult to parse the text file.

The name of the cache file is in the following format:
model={model}_split={split}_prompt-id={prompt_id}_progress={progress}.pkl
where
    {model} is the name of the model used to generate the predictions,
    {split} is the split of the dataset used to generate the predictions,
    {prompt_id} is a unique identifier for the prompt used to generate the
        predictions, and
    {progress} is the amount of predictions that have been generated so far.
"""

import logging
import os
import pickle
import re
from collections import defaultdict


# isort: off
from utils import pred_type, safeguard_filename

# isort: on


def determine_preds_cache_location(
    prompt_template: str, model: str, split: str, dir_caches: str
) -> str:
    """Determines the cache location where the predictions should be stored.

    If a cache file already exists for the given model, split, and prompt,
    that cache file is returned. Otherwise, the location where the predictions
    should be stored is returned. You can check whether the returned cache file
    location already exists by using `os.path.exists(cache_location)`.

    Args:
        prompt_template (str): The prompt template to use.
        model (str): The name of the model to evaluate.
        split (str): The split of the dataset to evaluate on.
        dir_caches (str): Path to the directory where the caches are stored.

    Returns:
        str: The location of the cache file.
    """
    logging.info("Determining cache location...")

    # Create the cache directory if it doesn't exist yet.
    os.makedirs(dir_caches, exist_ok=True)

    # Look for all files in the cache directory that match the model, split,
    # and prompt.
    safe_model = safeguard_filename(model)
    safe_split = safeguard_filename(split)
    # match.group(0) is the cache file name.
    # match.group(1) is prompt identifier.
    # match.group(2) is the amount of predictions that have been generated.
    matches_all = [
        re.fullmatch(
            f"model={safe_model}_split={safe_split}_"
            r"prompt-id=(\d+)_progress=(\d+)\.pkl",
            filename,
        )
        for filename in os.listdir(dir_caches)
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
            curr_location = os.path.join(dir_caches, match.group(0))
            with open(curr_location, "rb") as f:
                curr_prompt = pickle.load(f)["prompt"]
                if prev_prompt is None:
                    if curr_prompt == prompt_template:
                        cache_location = curr_location
                    prev_prompt = curr_prompt
                elif curr_prompt != prev_prompt:
                    raise ValueError(
                        "All cache files with the same prompt identifier "
                        "should contain the same prompt. This error should "
                        "never occur, unless a cache was manually modified.\n"
                        f"Prompt 1:\n'''\n{prev_prompt}\n'''\n"
                        f"was found in '{prev_location}', while\n"
                        f"Prompt 2:\n'''\n{curr_prompt}\n'''\n"
                        f"was found in '{curr_location}'."
                    )

    # If a cache file was found, return it.
    if cache_location is not None:
        return cache_location

    # No existing cache file was found.
    # Choose a prompt identifier that isn't already taken.
    prompt_id = 0
    while prompt_id in versions:
        prompt_id += 1

    # Return the new cache file name.
    return os.path.join(
        dir_caches,
        f"model={safe_model}_split={safe_split}_"
        f"prompt-id={prompt_id}_progress=0.pkl",
    )


def retrieve_from_cache(cache_location: str) -> list[pred_type]:
    """Retrieves previously generated predictions from a cache file.

    Args:
        cache_location (str): The location of the cache file.

    Returns:
        list[pred]: The previously generated predictions.
    """
    if os.path.isfile(cache_location):
        logging.info(f"Using cached predictions from {cache_location}.")
        with open(cache_location, "rb") as f:
            predictions = pickle.load(f)["predictions"]
        logging.info(f"Loaded {len(predictions)} predictions from cache.")
    else:
        logging.info("No cached predictions found.")
        predictions = []
    return predictions


def update_preds_cache(
    prompt_template: str,
    model: str,
    split: str,
    predictions: list[pred_type],
    cache_location: str,
) -> str:
    """Updates the specified cache file with the given predictions.

    Args:
        prompt_template (str): The prompt template to use.
        model (str): The name of the model to evaluate.
        split (str): The split of the dataset to evaluate on.
        predictions (list[pred]): The predictions to add to the cache.
        cache_location (str): The location of the cache file.

    Returns:
        str: The location of the updated cache file.
    """
    # Update the predictions cache file.
    new_cache_location = re.sub(
        r"progress=\d+", f"progress={len(predictions)}", cache_location
    )
    with open(new_cache_location, "wb") as f:
        pickle.dump(
            {
                "prompt": prompt_template,
                "model": model,
                "split": split,
                "predictions": predictions,
            },
            f,
        )

    # Delete the old cache file.
    if (
        new_cache_location != cache_location
        and os.path.isfile(new_cache_location)
        and os.path.isfile(cache_location)
    ):
        os.remove(cache_location)

    return new_cache_location
