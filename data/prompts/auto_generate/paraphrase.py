"""File that generates paraphrases and calculates their propensity score."""

import json

from betterprompt import calculate_perplexity
from tqdm import tqdm


# isort: off
from gpt3 import get_completion

# isort: on


def get_meta_prompts(seed_prompt: str) -> list[str]:
    """Get meta prompts for a given seed prompt.

    Meta prompts are used to generate the final prompts for the GPT-3 model.

    Args:
        seed_prompt (str): The seed prompt.

    Returns:
        list[str]: List of meta prompts.
    """
    return [
        (
            "Write a paraphrase for the following sentence.\n"
            f"Sentence: {seed_prompt}\nParaphrase: "
        ),
        (
            "Paraphrase the following sentence.\n"
            f"Sentence: {seed_prompt}\nParaphrase: "
        ),
        (
            "Write a likely paraphrase of the following text.\n"
            f"Text: {seed_prompt}\nParaphrase: "
        ),
        (
            "Write a similar sentence to the following one.\n"
            f"Sentence: {seed_prompt}\nSimilar sentence: "
        ),
        (
            "Write a variation of the following sentence.\n"
            f"Sentence: {seed_prompt}\nVariation: "
        ),
        (
            "How would you say the following sentence in a different way?\n"
            f"Sentence: {seed_prompt}\nDifferent way: "
        ),
    ]


def paraphrase_with_gpt(seed_prompts: list[str]) -> dict[str, list[str]]:
    """Generate a list of prompts for the GPT-3 model using the seed prompts.

    Args:
        seed_prompts (list[str]): List of seed prompts.

    Returns:
        dict[str, list[str]]: Dictionary of seed prompts mapping to their
            generated paraphrased versions.
    """
    # Initialize the dictionary.
    para_prompts_per_seed = {
        seed_prompt: [seed_prompt] for seed_prompt in seed_prompts
    }

    # Generate paraphrased prompts.
    for seed_prompt, meta_prompt, temperature in tqdm(
        [
            (seed_prompt, meta_prompt, temperature)
            for seed_prompt in seed_prompts
            for meta_prompt in get_meta_prompts(seed_prompt)
            for temperature in [0.5, 1.0, 1.5, 2.0]
        ]
    ):
        para_prompts_per_seed[seed_prompt].append(
            get_completion(
                prompt=meta_prompt,
                max_tokens=100,
                temperature=temperature,
                presence_penalty=-0.5,
            )
        )

    # Remove duplicates.
    for seed_prompt in para_prompts_per_seed:
        para_prompts_per_seed[seed_prompt] = list(
            set(para_prompts_per_seed[seed_prompt])
        )
    return dict(para_prompts_per_seed)


def rank_prompts(
    para_prompts_per_seed: dict[str, list[str]]
) -> dict[str, list[dict[str, float | str]]]:
    """Rank the prompts by their perplexity score.

    Args:
        para_prompts_per_seed (dict[str, list[str]]): Dictionary of seed
            prompts mapping to their generated paraphrased versions.

    Returns:
        dict[str, list[dict[str, float | str]]]: Dictionary of seed prompts
            mapping to their generated paraphrased versions together with the
            propensity score of these paraphrased versions, sorted by the
            perplexity score in ascending order.

            Example:
            {
                "<seed_prompt_1>": [
                    {
                        "perplexity": <perplexity_1>,
                        "para_prompt": "<para_prompt_1>",
                    },
                    {
                        "perplexity": <perplexity_2>,
                        "para_prompt": "<para_prompt_2>",
                    },
                    ...
                ],
                ...
            }
    """
    ranked_prompts = {}
    for seed_prompt, para_prompts in para_prompts_per_seed.items():
        ranked_prompts[seed_prompt] = sorted(
            (
                {
                    "perplexity": calculate_perplexity(para_prompt),
                    "para_prompt": para_prompt,
                }
                for para_prompt in para_prompts
                if len(para_prompt.split()) > 1
            ),
            key=lambda x: x["perplexity"],
        )
    return ranked_prompts


if __name__ == "__main__":
    seed_prompts = [
        ("Is the following post hateful?"),
        (
            "Could this post be considered offensive, disrespectful, or "
            "toxic to anyone/someone?"
        ),
    ]
    para_prompts_per_seed = paraphrase_with_gpt(seed_prompts)
    pretty_para_prompts_ranked = json.dumps(
        rank_prompts(para_prompts_per_seed), indent=4, default=str
    )
    print(pretty_para_prompts_ranked)
    with open("prompts/sorted_by_perplexity.json", "w") as f:
        f.write(pretty_para_prompts_ranked)
    print("Saved prompts to 'prompts/sorted_by_perplexity.json'.")
