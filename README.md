# Explainable Hate Speech Detection with Chain-of-Thought Reasoning and Post-Hoc Explanations
This paper is an assignment for the [Advanced Topics in Computational Semantics](https://cl-illc.github.io/semantics-2023/) course at the University of Amsterdam. We aim to explore the role of bias and stereotypes in online language, explicitly focusing on hate speech detection. We leverage large language models (LLMs) trained with in-context learning, enhanced by incorporating explanations into the training process. These explanations can be integrated as pre-answer reasoning or post-answer explanations, each offering different benefits.

Our research questions involve investigating whether we can prompt LLMs to reason about a potentially hateful post before or after making a prediction. We also examine the effectiveness of inductive versus deductive approaches on hate speech datasets, primarily using the SocialBiasFrames dataset. Our goal is to improve our understanding of hate speech detection in LLMs, ultimately contributing to developing more effective and accountable AI systems.

You can read our full report [here](https://github.com/SebastiaanJohn/explainable-hate-speech-detection/blob/main/ATCS_Report_2023.pdf).

## Requirements
The code is written in Python 3.10. The requirements can be installed using `pip install -r requirements.txt` or with the conda environment file `conda env create -f environment.yml`.

## Evaluation
The evaluation of the models is done using the `eval.py` script. The script takes the following arguments:

```bash
python3 src/eval.py --model <model> --prompt_name <name-of-prompt>
```

The model argument can be any of the T5 or GPT models from the [HuggingFace Transformers](https://huggingface.co/transformers/) library. The prompt name argument can be any of the following:

- `default_0shot_no_cot`: Default prompt for no CoT reasoning
- `default_0shot_cot_expl_first`: Default prompt for CoT reasoning with explanation generation first
- `default_4shot_cot_ans_first`: Default prompt 4-shot CoT reasoning with answer generation first
- `default_4shot_cot_expl_first`: Default prompt 4-shot CoT reasoning with explanation generation first

If you want to evaluate the model using a custom prompt, you can add the prompt to the `data/prompts` folder and use the name of the file as the prompt name argument.

By default, the generations of the models will be cached in the `data/caches` folder. If you happened to cancel the evaluation script, the next time you run it, it will use the cached generations instead of generating them again.

The model is evaluated on the SocialBiasFrames test set by default. You can change the split by adding the `--split` argument. The possible values are `train`, `validation`, and `test`. The dataset will be downloaded automatically on the first run.
