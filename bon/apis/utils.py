from typing import Iterable

import bon.utils.math_utils
from bon.data_models.inference import LLMResponse


def binary_response_logit(
    response: LLMResponse,
    tokens1: Iterable[str],
    tokens2: Iterable[str],
    token_idx: int = -1,
) -> float | None:
    """
    Returns logit(prob) that tokens1 is better than tokens2.
    We return the logit instead of the probability to avoid numerical instability.
    Returns None if the response was unable to be parsed.
    """
    assert response.logprobs is not None

    tokens1 = set(tokens1)
    tokens2 = set(tokens2)
    assert tokens1 & tokens2 == set()  # Check no overlap
    assert not any(" " in t for t in (tokens1 | tokens2))  # Check no spaces

    final_logprobs = response.logprobs[token_idx]
    final_tokens = final_logprobs.keys()
    if set(t.strip() for t in final_tokens) & (tokens1 | tokens2) == set():
        return None

    logprobs_1 = [final_logprobs[t] for t in final_tokens if t.strip() in tokens1]
    logprobs_2 = [final_logprobs[t] for t in final_tokens if t.strip() in tokens2]

    logprob_1 = bon.utils.math_utils.logsumexp(logprobs_1)
    logprob_2 = bon.utils.math_utils.logsumexp(logprobs_2)

    return logprob_1 - logprob_2


def get_combined_logprobs(
    response: LLMResponse,
    tokens1: Iterable[str],
    tokens2: Iterable[str],
    token_idx: int = -1,
) -> tuple[float | None, float | None]:
    """TODO: Consolidate with function above"""
    assert response.logprobs is not None

    tokens1 = set(tokens1)
    tokens2 = set(tokens2)
    assert tokens1 & tokens2 == set()  # Check no overlap
    assert not any(" " in t for t in (tokens1 | tokens2))  # Check no spaces

    final_logprobs = response.logprobs[token_idx]
    final_tokens = final_logprobs.keys()
    if set(t.strip() for t in final_tokens) & (tokens1 | tokens2) == set():
        return None, None

    logprobs_1 = [final_logprobs[t] for t in final_tokens if t.strip() in tokens1]
    logprobs_2 = [final_logprobs[t] for t in final_tokens if t.strip() in tokens2]

    logprob_1 = bon.utils.math_utils.logsumexp(logprobs_1)
    logprob_2 = bon.utils.math_utils.logsumexp(logprobs_2)

    return logprob_1, logprob_2
