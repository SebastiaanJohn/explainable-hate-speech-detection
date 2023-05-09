"""Functions to evaluate the generated prompts."""

from collections import defaultdict

import betterprompt
from gpt.gpt import get_completion


def get_meta_prompts(seed_prompt: str) -> list[str]:
    """Get meta prompts for a given seed prompt.

    These meta prompts are used to generate the final prompts for the GPT-3 model.

    Args:
        seed_prompt (str): The seed prompt.

    Returns:
        list[str]: List of meta prompts.
    """
    return [
        f"Write a paraphrase for the following sentence: {seed_prompt} Paraphrase:",
        f"{seed_prompt} Paraphrase:",
        f"Write a likely paraphrase of the text: {seed_prompt} Paraphrase:",
        f"Write a similar sentence similar to the following one: {seed_prompt} Paraphrase:",
        f"Paraphrase the following sentence: {seed_prompt} Paraphrase:",
        f"Write a variation of this sentence: {seed_prompt}",
        f"How would you say the following sentence in a different way? {seed_prompt}",
    ]


def paraphrase_with_gpt(seed_prompts: list[str]) -> dict[str, list[str]]:
    """Generate a list of prompts for the GPT-3 model using the seed prompts.

    Args:
        seed_prompts (list[str]): List of seed prompts.

    Returns:
        dict[str, list[str]]: Dictionary of seed prompts and their generated prompts.
    """
    paraphrased_prompts = defaultdict(list)
    for prompt in seed_prompts:
        meta_prompts = get_meta_prompts(prompt)
        for meta_prompt in meta_prompts:
            paraphrased_prompt = get_completion(
                prompt=meta_prompt,
                max_tokens=1000,
                temperature=0.7,
                presence_penalty=-0.5,
            )
            paraphrased_prompts[prompt].append(paraphrased_prompt)

    return dict(paraphrased_prompts)

def rank_prompts(
    paraphrased_prompts: dict[str, list[str]],
) -> dict[str, list[tuple[str, float]]]:
    """Rank the prompts by their perplexity score.

    #TODO: probably should combine all the prompts and rank them together.

    Args:
        paraphrased_prompts (dict[str, list[str]]): Dictionary of seed prompts and their
            generated prompts.

    Returns:
        dict[str, list[tuple[str, float]]]: Dictionary of seed prompts and their ranked

    Example:
        {
            "seed_prompt": [
                ("paraphrased_prompt_1", perplexity_score_1),
                ("paraphrased_prompt_2", perplexity_score_2),
                ...
            ]
        }
    """
    ranked_prompts = {}
    for seed_prompt, prompts in paraphrased_prompts.items():
        ranked_prompts[seed_prompt] = [
            (prompt, betterprompt.calculate_perplexity(prompt)) for prompt in prompts
        ]
        ranked_prompts[seed_prompt].sort(key=lambda x: x[1])

    return ranked_prompts



def rank_prompts_2(prompts: list[str]) -> list[tuple[str, float]]:
    """Rank the prompts by their perplexity score.

    Args:
        prompts (list[str]): List of prompts to rank.

    Returns:
        List of tuples containing the prompt and its perplexity score, sorted by
        the perplexity score in ascending order.

    Example:
        {
            "seed_prompt": [
                ("paraphrased_prompt_1", perplexity_score_1),
                ("paraphrased_prompt_2", perplexity_score_2),
                ...
            ]
        }
    """
    ranked_prompts = {}
    for prompt in prompts:
        ranked_prompts[prompt] = betterprompt.calculate_perplexity(prompt)

    return sorted(ranked_prompts.items(), key=lambda x: x[1])

if __name__ == "__main__":
    prompt = """Is the following post hateful? Answer with "yes" or "no". Do not explain your answer."""
    new_prompts = paraphrase_with_gpt([prompt])
    print(rank_prompts(new_prompts))
