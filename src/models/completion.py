"""Model pipeline for the project."""


from transformers import pipeline


def get_response(input_prompt: str, model: str) -> str:
    """Get response from model.

    Args:
        input_prompt (str): The input prompt to the model.
        model (str): The model checkpoint to use.

    Returns:
        str: The response from the model.
    """
    checkpoint = f"MBZUAI/{model}"
    model = pipeline("text2text-generation", model=checkpoint)

    return model(input_prompt, max_length=512, do_sample=True)[0][
        "generated_text"
    ]
