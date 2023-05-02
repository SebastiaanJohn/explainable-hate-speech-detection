"""GPT API wrapper."""

import logging
import os

import backoff
import openai


# get api key from environment variable

API_KEY = os.getenv("OPENAI_API_KEY")


@backoff.on_exception(
    backoff.expo,
    openai.error.RateLimitError,
    on_backoff=lambda details: logging.error(
        "API Error, %s", details["exception"]
    ),
    base=0.2,
)
@backoff.on_exception(
    backoff.expo,
    openai.error.ServiceUnavailableError,
    on_backoff=lambda details: logging.error(
        "API Error, %s", details["exception"]
    ),
    base=0.2,
)
@backoff.on_exception(
    backoff.expo,
    openai.error.APIError,
    on_backoff=lambda details: logging.error(
        "API Error, %s", details["exception"]
    ),
    base=0.2,
)
def get_completion(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 0,
    top_p: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> str:
    """Get completion from OpenAI API using GPT-3 model.

    Args:
        prompt (str): The prompt to complete.
        model (str, optional): The model to use. Defaults to settings.OPENAI_MODEL.
        temperature (int, optional): What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower
            values like 0.2 will make it more focused and deterministic.
        max_tokens (int, optional): Thee maximum number of tokens to generate in
            the completion.
            The token count of your prompt plus max_tokens cannot exceed the model's
            context length.
            Most models have a context length of 2048 tokens (except for the newest
            models, which support 4096).
        top_p (float, optional): An alternative to sampling with temperature,
            called nucleus sampling, where the model considers the results of the
            tokens with top_p probability mass. So 0.1 means only the tokens comprising
            the top 10% probability mass are considered.
        frequency_penalty (float, optional): Number between -2.0 and 2.0.
            Positive values penalize new tokens based on their existing frequency
            in the text so far, decreasing the model's likelihood to repeat
            the same line verbatim.
        presence_penalty (float, optional): Number between -2.0 and 2.0. Positive
            values penalize new tokens based on whether they appear in the text so far,
            increasing the model's likelihood to talk about new topics.

    Returns:
        str: The completion generated by the model.
    """
    openai.api_key = API_KEY
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

    return response["choices"][0]["text"].strip()


@backoff.on_exception(
    backoff.expo,
    openai.error.RateLimitError,
    on_backoff=lambda details: print(f"API Error, {details['exception']}"),
    base=0.2,
)
@backoff.on_exception(
    backoff.expo,
    openai.error.ServiceUnavailableError,
    on_backoff=lambda details: print(f"API Error, {details['exception']}"),
    base=0.2,
)
@backoff.on_exception(
    backoff.expo,
    openai.error.APIError,
    on_backoff=lambda details: print(f"API Error, {details['exception']}"),
    base=0.2,
)
def get_chat_completion(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 0,
    top_p: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> str:
    """Get chat completion from OpenAI API using GPT-3 model.

    Args:
        messages (list[dict[str, str]]): The user and system messages to complete.
        model (str, optional): The model to use. Defaults to settings.OPENAI_MODEL.
        temperature (int, optional): What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower
            values like 0.2 will make it more focused and deterministic.
        max_tokens (int, optional): Thee maximum number of tokens to generate in
            the completion.
            The token count of your prompt plus max_tokens cannot exceed the model's
            context length.
            Most models have a context length of 2048 tokens (except for the newest
            models, which support 4096).
        top_p (float, optional): An alternative to sampling with temperature,
            called nucleus sampling, where the model considers the results of the
            tokens with top_p probability mass. So 0.1 means only the tokens comprising
            the top 10% probability mass are considered.
        frequency_penalty (float, optional): Number between -2.0 and 2.0.
            Positive values penalize new tokens based on their existing frequency
            in the text so far, decreasing the model's likelihood to repeat
            the same line verbatim.
        presence_penalty (float, optional): Number between -2.0 and 2.0. Positive
            values penalize new tokens based on whether they appear in the text so far,
            increasing the model's likelihood to talk about new topics.

    Returns:
        str: The completion generated by the model.
    """
    openai.api_key = API_KEY
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=messages,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    logging.debug(response["usage"])

    return response["choices"][0]["message"]["content"]
