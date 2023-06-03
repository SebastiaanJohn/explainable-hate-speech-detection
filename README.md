# Integrating Chain-of-Thought Reasoning and Explainable AI for Enhanced Hate Speech Detection and Explanation Generation in Large Language Models

This repository investigates the integration of Chain-of-Thought (CoT) reasoning and Explainable AI (XAI) techniques for enhancing hate speech detection and explanation generation in large language models (LLMs). We aim to prompt LLMs to reason about potentially hateful content and produce a prediction or explanation, comparing inductive and deductive approaches for improved performance on hate speech datasets with full-text annotations. The primary dataset utilized for this project is SocialBiasFrames (Sap et al. 2019). We aim to provide human content moderators with more comprehensive and understandable explanations for detected hate speech, ultimately improving the moderation process and increasing the transparency of AI-driven decisions.

## Requirements

The code is written in Python 3.10. The requirements can be installed using `pip install -r requirements.txt` or with the conda environment file `conda env create -f environment.yml`.

## Evaluation

The evaluation of the models is done using the `eval.py` script. The script takes the following arguments:

```bash
python src/eval.py --model <model> --prompt_name <name-of-prompt> 
```

The model argument can be any of the T5 or GPT models from the [HuggingFace Transformers](https://huggingface.co/transformers/) library. The prompt name argument can be any of the following: 

- `default_0shot_no_cot`: Default prompt for no CoT reasoning
- `default_0shot_cot_expl_first`: Default prompt for CoT reasoning with explanation generation first
- `default_4shot_cot_ans_first`: Default prompt 4-shot CoT reasoning with answer generation first
- `default_4shot_cot_expl_first`: Default prompt 4-shot CoT reasoning with explanation generation first

If you want to evaluate the model using a custom prompt, you can add the prompt to the `data/prompts` folder and use the name of the file as the prompt name argument.

By default, the generations of the models will be cached in the `data/caches` folder. If you happened to cancel the evaluation script, the next time you run it, it will use the cached generations instead of generating them again.

The model is evaluated on the SocialBiasFrames test set by default. You can change the split by adding the `--split` argument. The possible values are `train`, `validation`, and `test`. The dataset will be downloaded automatically on the first run.

