import torch
import transformers
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
)


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

if __name__ == "__main__":
    import time
    model_name = "MBZUAI/LaMini-T5-738M"
    device = torch.device("mps")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    input_text = "The quick brown fox jumps over the lazy dog, yes?"
    output_choices = ["Yes", "No",]

    probs = get_token_probability_distribution(
        input_text,
        output_choices,
        tokenizer,
        device,
        model,
    )
    print("Probabilities:", probs)

    print("Input text:", input_text)

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate the first prediction.
    start = time.time()
    first_prediction_response = model.generate(
        input_ids,
        max_length=30,
        num_beams=3,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    end = time.time()
    print("Time to generate first prediction:", end - start)
    print("First prediction:", tokenizer.decode(first_prediction_response[0], skip_special_tokens=True))
