from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-61M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-61M")

prompt = 'RT @_LexC__: I\'m convinced that some of y\'all bitches get pregnant purposely because "birth control &amp; plan b pills" are effective &#128533;&#128056;&#9749;&#65039;'
sentence = f"{prompt}\nIs this post hateful? (yes/no):"
input_ids = tokenizer(sentence, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids,
    max_new_tokens=20,
    num_beams=4,
    num_return_sequences=1,
    return_dict_in_generate=True,
    output_scores=True
    )

transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
)

text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

