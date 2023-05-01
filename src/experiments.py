from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-61M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-61M")

def get_label_token_ids(tokenizer) -> dict:
    """Returns the token ids for each label."""
    label_to_token = {}
    for label in ["yes", "Yes", "YES", "no", "No", "NO"]:
        label_to_token[label] = tokenizer(label, return_tensors="pt").input_ids[0][0]

    return label_to_token

label_tokens = get_label_token_ids(tokenizer)
print(label_tokens)

prompt = 'RT @_LexC__: I\'m convinced that some of y\'all bitches get pregnant purposely because "birth control &amp; plan b pills" are effective &#128533;&#128056;&#9749;&#65039;'
sentence = f"{prompt}\nIs this post hateful?"
input_ids = tokenizer(sentence, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids,
    max_new_tokens=20,
    num_beams=4,
    num_return_sequences=1,
    return_dict_in_generate=True,
    output_scores=True
    )

print(len(outputs.scores))
print(outputs.scores[19].shape)
print(outputs.sequences)


transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
)

text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

print(text)
