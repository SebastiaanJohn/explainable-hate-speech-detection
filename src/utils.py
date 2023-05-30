"""Utility functions for generating or loading model predictions."""

import logging
import os
import pickle
import re
from collections import defaultdict

import torch
import transformers
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
)


PROMPTS_DIR = "./prompts"
PREDS_CACHE_DIR = "./data/preds_cache"
VALID_FILENAME_CHARS = (
    r"-=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def safeguard_filename(filename: str) -> str:
    """Converts a filename to a safe filename by removing special characters.

    Args:
        filename (str): The filename to convert.

    Returns:
        str: The converted filename.
    """
    return "".join(c if c in VALID_FILENAME_CHARS else "-" for c in filename)


def determine_preds_cache_location(
    model: str, split: str, prompt: str, preds_cache_dir: str = PREDS_CACHE_DIR
) -> str:
    """Determines the cache location where the predictions should be stored.

    If a cache file already exists for the given model, split, and prompt,
    that cache file is returned. Otherwise, the location where the predictions
    should be stored is returned. You can check whether the returned cache file
    location already exists by using `os.path.exists(cache_location)`.

    Args:
        model (str): The name of the model.
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
    # match.group(0) is the cache file name.
    # match.group(1) is prompt identifier.
    # match.group(2) is the amount of predictions that have been generated.
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
    """Extracts information from the given cache file.

    Args:
        cache_location (str): The location of the cache file.

    Returns:
        Tuple containing:
            model (str): The name of the model.
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


def construct_full_prompt(prompt: str, post: str) -> str:
    """Constructs a full prompt from the given prompt and post.

    Args:
        prompt (str): The prompt to use.
        post (str): The post to use.

    Returns:
        str: The full prompt.
    """
    return prompt.replace("${post}", post)


def classify(prediction: str, labels: set[str]) -> str:
    """Extract a label from the response.

    Args:
        response (str): The response from the model.
        labels (set[str]): The set of possible ground truth labels.

    Returns:
        str: A label in the given set of labels, or an alternative string if
            a fitting label could not be found. These alternative strings can
            be useful for getting insight into the model's outputs.
    """
    prediction = prediction.lower()

    # Check which labels the prediction contains.
    labels_in_prediction = set()
    for label in labels:
        if re.search(rf"\b{label.lower()}\b", prediction.lower()) is not None:
            labels_in_prediction.add(label)

    # Single label in prediction.
    if len(labels_in_prediction) == 1:
        return labels_in_prediction.pop()

    # Multiple labels in prediction.
    if len(labels_in_prediction) > 1:
        return "/".join(sorted(labels_in_prediction))

    # No labels in prediction.
    return "None"


def log_pred(
    example_idx: int,
    full_prompt: str,
    prediction: str,
    label_true: str,
    label_pred: str,
) -> None:
    """Show a single prediction.

    Args:
        example_idx (int): The index of the example.
        full_prompt (str): The full prompt, including the post.
        prediction (str): The model's prediction.
        label_true (str): The true label.
        label_pred (str): The predicted label.
    """
    logging.info(f" Example {example_idx} ".center(65, "-"))
    full_prompt = full_prompt.rstrip("\n")
    logging.info("\n>>> ".join(["Input prompt:"] + full_prompt.split("\n")))
    logging.info("\n>>> ".join(["Model's output:"] + prediction.split("\n")))
    logging.info(f"True label: {label_true}")
    logging.info(f"Predicted label: {label_pred}")
    logging.info("")


def update_preds_cache(
    model: str,
    split: str,
    prompt: str,
    predictions: list[str],
    cache_location: str,
) -> str:
    """Updates the predictions cache file with the given predictions.

    Args:
        model (str): The name of the model.
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

def get_token_probability_distribution(
    input_text: str,
    output_choices: list[str],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    model: transformers.PreTrainedModel,
) -> list[tuple[str, float]]:
    """For a single input text, returns the probability distribution over possible next tokens considering only the given list of output choices.

    Args:
        input_text (str): the input text
        output_choices (list[str]): the allowed output choices (must correspond to single tokens in the model's vocabulary)
        tokenizer (PreTrainedTokenizer): the tokenizer for the model
        device (torch.device): the device to run the model on
        model (transformers.PreTrainedModel): the model

    Returns:
        a list, where output[j] is a (output, probability) tuple for the jth output choice.
    """
    # Convert output choices to token ids
    output_choice_ids = [
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0] for word in output_choices
    ]

    # Encode input_text and run it through the model
    if isinstance(model, T5ForConditionalGeneration):
        # T5 model
        inputs = tokenizer.encode_plus(input_text, return_tensors="pt").to(device)
        inputs['decoder_input_ids'] = tokenizer.encode(tokenizer.pad_token, return_tensors='pt').to(device)
        logits = model(**inputs)["logits"][0]
    elif isinstance(model, GPT2LMHeadModel):
        # GPT-2 model
        inputs = tokenizer.encode_plus(input_text, return_tensors="pt").to(device)
        logits = model(**inputs).logits[0]
    else:
        raise ValueError("Unsupported model type.")

    # Calculate probabilities
    output_probabilities = torch.nn.functional.softmax(logits, dim=-1)
    output_probabilities = output_probabilities[0, output_choice_ids]

    # Re-normalize probabilities so they sum to 1
    output_probabilities = output_probabilities / output_probabilities.sum()

    # Pair choices with probabilities
    choices_with_probabilities = list(zip(output_choices, output_probabilities.tolist()))

    return choices_with_probabilities

def generate_predictions(
    model: str,
    split: str,
    prompt: str,
    dataset: Dataset,
    min_preds: int = 0,
    show_preds: int = 0,
    preds_cache_dir: str = PREDS_CACHE_DIR,
    save_every: int = 1000,
    max_length: int = 200,
    chain_of_thought: bool = False,
    second_prompt: str = "",
    use_mps: bool = False,
    output_choices: list[str] = ["Yes", "No"],
) -> list[str]:
    """Generates predictions using the given model.

    WARNING: The generated list may be larger or smaller than `min_preds`,
        which respectively depends on the amount of cached predictions and the
        size of the dataset.

    Args:
        model (str): The name of the model.
        split (str): The split of the dataset to generate predictions for. Must
            be one of "train", "validation", or "test".
        prompt (str): The prompt to be prepended to each post.
        dataset (Dataset): The dataset to generate predictions for.
        min_preds (int, optional): The minimum amount of predictions that
            should be generated. If the cache file already contains some of
            these predictions, the model will continue generating predictions
            until the minimum amount is reached. If 0, all predictions will be
            generated. Defaults to 0.
        show_preds (int, optional): The number of generated predictions that
            should be shown. If 0, no predictions will be shown. Defaults to 0.
        preds_cache_dir (str, optional): The directory where the cache files
            are stored. Defaults to "./data/preds_cache".
        save_every (int, optional): The amount of predictions that should be
            generated before saving them to the cache file. Defaults to 1000.
        max_length (int, optional): The maximum length of the generated
            predictions. Defaults to 200.
        chain_of_thought (bool, optional): Whether the model should be
            conditioned on the previous prediction. Defaults to False.
        second_prompt (str, optional): The prompt to be prepended to each
            post, which is used when `chain_of_thought` is True. Defaults to
            "".
        use_mps (bool, optional): Whether the model should train on Apple GPU's.
            Defaults to False.
        output_choices (list[str], optional): The output choices to use when
            generating predictions. Defaults to ["Yes", "No"].

    Returns:
        list[str]: The predictions generated by the model.
    """
    # Determine the cache location.
    # The predictions are retrieved from a cache file if it already exists.
    # This cache file is a pickle file that contains a dictionary with:
    # - "prompt": The prompt used to generate the predictions.
    # - "predictions": A list of prediction strings.
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
    labels = set(dataset["offensiveYN"])
    dataset = dataset.select(range(*select))

    # Get the right device.
    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load tokenizer and model.
    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model).to(device)

    # Construct the full prompt for each example.
    if "${post}" not in prompt:
        raise ValueError(
            "The prompt must contain the string '${post}' to indicate where "
            "the post should be inserted."
        )
    full_prompts = [
        construct_full_prompt(prompt, example["post"]) for example in dataset
    ]

   # Generate the (missing) predictions.
    logging.info("Generating predictions...")
    for i, (example, full_prompt) in enumerate(
        tqdm(
            list(zip(dataset, full_prompts)),
            desc="Generating predictions",
            unit="preds",
            initial=select[0],
            total=select[1],
            )
        ):
        # ================ Debugging ================
        print("===================prompt==================")
        print(full_prompt)
        # ================ Debugging ================
        if chain_of_thought:
            input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

            # Generate the first prediction.
            first_prediction_response = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=3,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

            # Extract the model's output prediction from the first response.
            first_prediction = tokenizer.decode(
                first_prediction_response[0], skip_special_tokens=True
            ).strip()

            print("===================first_prediction==================")
            print(first_prediction)

            # Construct the second prompt with the first prediction.
            new_prompt = full_prompt + first_prediction + f"\n{second_prompt}"

            print("===================full_prompt==================")
            print(new_prompt)
        else:
            new_prompt = full_prompt

        # Generate the prediction.
        probabilities = get_token_probability_distribution(
            new_prompt,
            output_choices,
            tokenizer,
            device,
            model,
        )
        print("===================probabilities==================")
        print(probabilities)
        final_prediction = sorted(probabilities, key=lambda x: x[1], reverse=True)[0][0]
        print("===================final_prediction==================")
        print(final_prediction)
        print("===================example[offensiveYN]==============")
        print(example["offensiveYN"])
        print("=====================================================")
        predictions.append(final_prediction)

        # Show a log of the prediction if requested.
        if i < select[0] + show_preds:
            log_pred(
                i,
                second_prompt,
                final_prediction,
                example["offensiveYN"],
                classify(final_prediction, labels),
            )
        else:
            predictions.append(first_prediction)

            # Show a log of the prediction if requested.
            if i < select[0] + show_preds:
                log_pred(
                    i,
                    full_prompt,
                    first_prediction,
                    example["offensiveYN"],
                    classify(first_prediction, labels),
                )

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
    """Loads the predictions generated by the given model.

    Args:
        model (str): The name of the model.
        split (str): The split of the dataset to generate predictions for. Must
            be one of "train", "validation", or "test".
        prompt (str): The prompt to be prepended to each post.
        preds_cache_dir (str, optional): The directory where the cache files
            are stored. Defaults to "./data/preds_cache".

    Returns:
        list[str]: The predictions generated by the model.
    """
    # Determine the cache location.
    # The predictions are retrieved from a cache file if it already exists.
    # This cache file is a pickle file that contains a dictionary with:
    # - "prompt": The prompt used to generate the predictions.
    # - "predictions": A list of prediction strings.
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

    raise FileNotFoundError(f"No cached predictions found at {cache_location}.")
