"""This file contains functions for generating predictions from a model."""

import logging

import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
)


# isort: off
from utils import pred_type, log_prediction
from generation.cache import (
    determine_preds_cache_location,
    retrieve_from_cache,
    update_preds_cache,
)

# isort: on


def get_probability_positive(
    prompt: str,
    labels_positive: list[str],
    labels_negative: list[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
) -> float:
    """Get the probability of the next token being a positive label.

    Args:
        prompt (str): The input prompt to use.
        labels_positive (list[str]): Labels that are considered positive.
        labels_negative (list[str]): Labels that are considered negative.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        model (PreTrainedModel): The model to use.
        device (torch.device): The device to use.

    Returns:
        float: The probability of the next token being a positive label.
    """
    # Encode prompt and run it through the model.
    with torch.no_grad():
        if isinstance(model, T5ForConditionalGeneration):
            # T5 and Flan-T5 models.
            inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}
            inputs["decoder_input_ids"] = tokenizer.encode(
                tokenizer.pad_token, return_tensors="pt"
            ).to(device)
            logits = model(**inputs)["logits"][0, 0, :]
        elif isinstance(model, GPT2LMHeadModel):
            # Cerebras and GPT models.
            inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}
            output_index = inputs["attention_mask"].sum() - 1
            logits = model(**inputs)["logits"][0, output_index, :]
        elif isinstance(model, GPTNeoForCausalLM):
            # Neo models.
            raise NotImplementedError("GPT-Neo models are not yet supported.")
            # TODO Add support for GPT-Neo models.
        else:
            raise ValueError("Unsupported model type.")

    # Convert labels to token ids.
    label_ids = [
        tokenizer.encode(token)[0]
        for token in labels_positive + labels_negative
    ]

    # Calculate probabilities.
    logging.debug(f"{inputs['input_ids'].shape=}")
    logging.debug(f"{logits.shape=}")
    probs_labels = logits[label_ids]

    top_tokens = torch.topk(logits, 15).indices.tolist()
    top_tokens = [tokenizer.decode([token]) for token in top_tokens]
    logging.debug(f"Top tokens: {top_tokens}")

    # Normalize probabilities so they sum to 1.
    probs_labels = F.softmax(probs_labels, dim=-1)
    logging.debug(f"{probs_labels=}")

    # Calculate probability of the next token being a positive label.
    probs_positive = probs_labels[: len(labels_positive)].sum()

    return probs_positive.item()


def generate_predictions(
    dataset: Dataset,
    prompt_template: str,
    model: str,
    split: str,
    max_length: int,
    save_every: int,
    preds_to_gen: int,
    preds_to_show: int,
    labels_positive: list[str],
    labels_negative: list[str],
    use_mps: bool,
    dir_caches: str,
) -> list[pred_type]:
    """Generates predictions using the given model.

    WARNING: The generated list may be larger or smaller than `preds_to_gen`,
        which respectively depends on the amount of cached predictions and the
        size of the dataset.

    Args:
        dataset (Dataset): The dataset to generate predictions for.
        prompt_template (str): The prompt template to use.
        model (str): The name of the model to evaluate.
        split (str): The split of the dataset to evaluate on.
        max_length (int): Maximum amount of tokens the model can generate for
            its explanation.
        save_every (int): Number of predictions to generate before caching
            them.
        preds_to_gen (int): Minimum number of predictions to generate. If the
            cache file already contains some of these predictions, the model
            will continue generating predictions until this number is reached.
            If 0, all predictions will be generated.
        preds_to_show (int): Number of generated predictions that should be
            logged.
        labels_positive (list[str]): Labels that are considered positive.
        labels_negative (list[str]): Labels that are considered negative.
        use_mps (bool): If True, the model is evaluated on an Apple GPU.
        dir_caches (str): Path to the directory where the caches are stored.

    Returns:
        list[float]: For every sample, the model's confidence that it should
            be classified as positive.
    """
    # Determine the cache location.
    cache_location = determine_preds_cache_location(
        prompt_template, model, split, dir_caches
    )

    # Load existing predictions from the cache file.
    predictions = retrieve_from_cache(cache_location)

    # Select a subset of the dataset to generate predictions for.
    if preds_to_gen > dataset.num_rows:
        logging.warning(
            f"The minimum number of predictions requested ({preds_to_gen}) is "
            f"greater than the size of the dataset ({dataset.num_rows}). "
            "It will be decreased accordingly."
        )
    select = (
        len(predictions),
        dataset.num_rows
        if preds_to_gen == 0 or preds_to_gen > dataset.num_rows
        else preds_to_gen,
    )
    if select[0] >= select[1]:
        return predictions
    logging.info(
        f"Generating predictions for examples {select[0]} to {select[1]}."
    )
    dataset = dataset.select(range(*select))

    # Get a reference to the device used for all computations that follow.
    if use_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Load the tokenizer and the model.
    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    logging.info(f"Using tokenizer: {type(tokenizer).__name__}")
    if "-T5-" in model:
        # T5 and Flan-T5 models.
        model = T5ForConditionalGeneration.from_pretrained(model).to(device)
    elif "-Cerebras-" in model or "-GPT-" in model:
        # Cerebras and GPT models.
        model = GPT2LMHeadModel.from_pretrained(model).to(device)
    elif "-Neo-" in model:
        # Neo models.
        model = GPTNeoForCausalLM.from_pretrained(model).to(device)
    else:
        raise ValueError("Unsupported model type.")
    model.eval()
    logging.info(f"Using model: {type(model).__name__}")

    # Pre-compute the locations of some of the special tokens.
    answer_location = prompt_template.find("{answer}")
    explanation_location = prompt_template.find("{explanation}")
    output_ids = None

    # Generate predictions.
    logging.info("Generating predictions...")
    for example_idx, example in enumerate(
        tqdm(
            dataset,
            desc="Generating predictions",
            unit="preds",
            initial=select[0],
            total=select[1],
        )
    ):
        if explanation_location != -1:
            # Construct the prompt.
            prompt = prompt_template[:explanation_location].format(
                post=example["post"]
            )

            # Encode the prompt and run it through the model.
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                device
            )

            # Generate the model's explanation.
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=len(input_ids) + max_length,
                    num_beams=3,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )

            # Extract the model's output prediction from the first response.
            explanation = tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )

            if isinstance(model, GPT2LMHeadModel):
                # Remove the prompt from the explanation.
                explanation = explanation[len(prompt) :]

            # Strip the explanation of any trailing whitespace.
            explanation = explanation.strip()

            # Construct the prompt.
            prompt = prompt_template[:answer_location].format(
                post=example["post"], explanation=explanation
            )
        else:
            # Construct the prompt.
            prompt = prompt_template[:answer_location].format(
                post=example["post"]
            )

        # Generate the probabilities for each label.
        prob_positive = get_probability_positive(
            prompt, labels_positive, labels_negative, tokenizer, model, device
        )

        if explanation_location != -1:
            predictions.append(
                {"explanation": explanation, "prob_positive": prob_positive}
            )
        else:
            predictions.append({"prob_positive": prob_positive})

        # Show a log of the prediction if requested.
        if example_idx < select[0] + preds_to_show:
            log_prediction(
                example_idx,
                example["post"],
                example["offensiveYN"],
                prob_positive,
                explanation=explanation if explanation_location != -1 else "",
            )

        # To prevent crashes or other errors from causing the loss of all
        # computed predictions, we save the predictions to the cache file after
        # every `save_every` predictions.
        if len(predictions) % save_every == 0:
            cache_location = update_preds_cache(
                prompt_template, model, split, predictions, cache_location
            )

    # Update the cache file with the final predictions.
    update_preds_cache(
        prompt_template, model, split, predictions, cache_location
    )

    return predictions
