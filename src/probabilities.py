


def query_model_batch(self, input_texts: list[str]):
    assert all('<extra_id_0>' in input_text for input_text in input_texts)
    output_texts = ['<extra_id_0>'] * len(input_texts)
    inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
    inputs = {key: val.to(self._device) for key, val in inputs.items()}
    output_ids = self._tokenizer.batch_encode_plus(output_texts, return_tensors='pt')['input_ids'].to(self._device)
    return self._model(labels=output_ids, **inputs)['logits'][:, 1, :]

def get_token_probability_distribution(
    input_texts: list[str],
    output_choices: list[str],
    tokenizer,
    device,
) -> list[list[tuple[str, float]]]:
    """For a batch of input texts, returns the probability distribution over possible next tokens considering only the given list of output choices.

    Args:
        input_texts: the input texts
        output_choices: the allowed output choices (must correspond to single tokens in the model's vocabulary)

    Returns:
        a list of lists, where output[i][j] is a (output, probability) tuple for the ith input and jth output choice.
    """
    output_choice_ids = []
    for word in output_choices:
        tokens = tokenizer.tokenize(word)
        token_id = tokenizer.convert_tokens_to_ids(tokens)[0]
        output_choice_ids.append(token_id)

    logits = query_model_batch(input_texts)
    result = []

    for idx, _ in enumerate(input_texts):
        output_probabilities = logits[idx][output_choice_ids].softmax(dim=0)
        choices_with_probabilities = list(zip(output_choices, (prob.item() for prob in output_probabilities)))
        result.append(choices_with_probabilities)

    return result

