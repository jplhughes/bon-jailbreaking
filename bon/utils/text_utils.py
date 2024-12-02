import random
import re
from functools import lru_cache
from typing import Optional

import pydantic
import tiktoken


class AttackString(pydantic.BaseModel):
    token_ids: list[int]

    def decode(self, tokenizer: tiktoken.core.Encoding) -> str:
        return tokenizer.decode(self.token_ids)

    def get_normalised_string(self, tokenizer: tiktoken.core.Encoding) -> str:
        return " ".join(self.decode(tokenizer).split())


@lru_cache()
def get_tokenizer() -> tiktoken.core.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def get_filtered_token_ids(tokenizer: tiktoken.core.Encoding, regex_pattern: Optional[str] = None) -> list[int]:
    """Get token IDs optionally filtered by a regex pattern.

    Args:
        tokenizer: The tiktoken tokenizer to use
        regex_pattern: Optional regex pattern to filter tokens. If None, returns all valid token IDs.

    Returns:
        List of token IDs matching the filter criteria
    """
    # Special ids such as <|endoftext|> which we want to avoid sampling
    special_ids = {
        tokenizer.encode(t, allowed_special=tokenizer.special_tokens_set)[0] for t in tokenizer.special_tokens_set
    }

    # These cause the tokenizer to crash due to pyo3_runtime.PanicException
    error_ids = set(range(100261, 100276)) | {100256}

    # 0 is our initialisation so we want to avoid that
    disallowed = special_ids | error_ids | {0}
    ids_to_sample = set(range(tokenizer.n_vocab)) - disallowed

    tokens = [(id, tokenizer.decode([id])) for id in ids_to_sample]

    if regex_pattern is None:
        return [id for id, _ in tokens]

    return [id for id, token in tokens if re.match(regex_pattern, token)]


def get_attack_string(num_tokens: int) -> AttackString:
    tokenizer = get_tokenizer()
    all_ids = get_filtered_token_ids(tokenizer)
    attack_string = AttackString(token_ids=random.sample(all_ids, num_tokens))
    return attack_string
