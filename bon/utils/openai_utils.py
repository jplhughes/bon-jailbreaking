import logging

import numpy as np
import openai
import openai.types
import openai.types.chat
import openai.types.chat.chat_completion
import scipy.special
import tiktoken

from bon.utils import utils


def print_choice(
    choice: openai.types.chat.chat_completion.Choice,
    logprob_dict: dict[int, tuple[float, str, str]] | None = None,
    width: int = 80,
):
    """
    Prints a CompletionChoice returned by the OpenAI API in a nice format.

    Includes both the text of the choice, as well as the logprobs of the choice.
    If logprob_dict (returned by get_top_chat_logprobs) is provided, then
    logprobs are taken from this dict instead of the choice object itself.

    width specifies the line width to use when printing the text of the choice.
    """

    utils.print_with_wrap(choice.message.content, width=width)
    print()
    if logprob_dict is not None:
        for logprob, token, _ in logprob_dict.values():
            log10prob = logprob / np.log(10)
            prob = np.exp(logprob)
            token = token.encode("unicode_escape").decode()
            print(f"prob: {prob:.6f}, log10prob:{log10prob:>8.4f}, token: '{token}'")
    else:
        for top_logprob in choice.logprobs.content[0].top_logprobs:
            logprob = top_logprob.logprob
            log10prob = logprob / np.log(10)
            prob = np.exp(logprob)
            token = top_logprob.token.encode("unicode_escape").decode()
            print(f"prob: {prob:.6f}, log10prob:{log10prob:>8.4f}, token: '{token}'")


def get_top_chat_logprobs(
    model: str,
    messages: list[dict[str, str]],
    seed: int = 42,
    n_logprobs: int = 20,
) -> dict[int, tuple[float, str, str]]:
    """
    Returns a dict mapping token_idx to (logprob, token_str, system_fingerprint)
    for the top n_logprobs tokens.

    Supports a maximum of 305 logprobs, but is flaky for 290+ logprobs
    (i.e. 290-305 logprobs will sometimes work, sometimes error). 306 logprobs
    and above will always error (unless OpenAI API changes).

    WARNING: Probably does not handle STOP tokens properly. You should check
             over the logic in this function if you want to accurately measure
             the probability STOP tokens are generated.
    """
    if model == "gpt-4-0125-preview":
        logging.warning("This function has issues with gpt-4-0125-preview.")

    tokenizer = tiktoken.encoding_for_model(model)
    assert 1 <= n_logprobs <= tokenizer.n_vocab

    client = openai.Client()

    def query_api(**kwargs):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1,
            n=1,
            logprobs=True,
            top_logprobs=5,
            seed=seed,
            stop=[],
            **kwargs,
        )

    base_resp = query_api()

    # Maps token_idx bytes to (logprob, token_str).
    logprob_dict: dict[int, tuple[float, str, str]] = {
        tokenizer.encode_single_token(bytes(top_logprobs.bytes)): (
            top_logprobs.logprob,
            top_logprobs.token,
            base_resp.system_fingerprint,
        )
        for top_logprobs in base_resp.choices[0].logprobs.content[0].top_logprobs
    }

    BIAS = -100
    while len(logprob_dict) < n_logprobs:
        log_masked_sum = scipy.special.logsumexp([logprob for logprob, _, _ in logprob_dict.values()])
        unmasked_sum = -scipy.special.expm1(log_masked_sum)
        log_unmasked_sum = np.log(unmasked_sum)

        resp = query_api(logit_bias={token_idx: BIAS for token_idx in logprob_dict.keys()})
        for top_logprob in resp.choices[0].logprobs.content[0].top_logprobs:
            if len(logprob_dict) >= n_logprobs:
                break

            token_str = top_logprob.token
            if token_str in ["<|end|>", "<|endoftext|>"]:
                token_idx = tokenizer.eot_token
            else:
                token_idx = tokenizer.encode_single_token(bytes(top_logprob.bytes))

            biased_logprob = top_logprob.logprob
            true_logprob = biased_logprob + np.logaddexp(log_masked_sum + BIAS, log_unmasked_sum)

            logprob_dict[token_idx] = (
                true_logprob,
                token_str,
                resp.system_fingerprint,
            )

    return logprob_dict
