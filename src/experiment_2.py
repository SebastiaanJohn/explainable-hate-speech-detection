import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-61M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-61M")

prompt = 'RT @_LexC__: I\'m convinced that some of y\'all bitches get pregnant purposely because "birth control &amp; plan b pills" are effective &#128533;&#128056;&#9749;&#65039;'
sentence = f"{prompt}\nIs this post hateful? (yes/no):"

# Generate input_ids and decoder_input_ids for 'yes' and 'no' answers
input_ids_yes = tokenizer(sentence + "yes", return_tensors="pt").input_ids
input_ids_no = tokenizer(sentence + "no", return_tensors="pt").input_ids
decoder_input_ids_yes = tokenizer("yes", return_tensors="pt").input_ids
decoder_input_ids_no = tokenizer("no", return_tensors="pt").input_ids

with torch.no_grad():
    logits_yes = model(input_ids_yes, decoder_input_ids=decoder_input_ids_yes).logits[:, -1, :]
    logits_no = model(input_ids_no, decoder_input_ids=decoder_input_ids_no).logits[:, -1, :]

# Calculate the likelihood of 'yes' and 'no'
likelihood_yes = torch.exp(logits_yes[:, tokenizer.convert_tokens_to_ids('yes')]).item()
likelihood_no = torch.exp(logits_no[:, tokenizer.convert_tokens_to_ids('no')]).item()

print(likelihood_yes, likelihood_no)

# Classify the post based on the higher likelihood, and convert the answer to lowercase
classification = "yes" if likelihood_yes > likelihood_no else "no"
generated_answer = classification.lower()
print(generated_answer)
