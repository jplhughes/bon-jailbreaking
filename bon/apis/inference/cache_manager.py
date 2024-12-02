import logging
from pathlib import Path
from typing import List, Tuple, Union

import filelock

from bon.data_models import (
    BatchPrompt,
    EmbeddingParams,
    EmbeddingResponseBase64,
    LLMCache,
    LLMCacheModeration,
    LLMParams,
    LLMResponse,
    Prompt,
    TaggedModeration,
)
from bon.data_models.hashable import deterministic_hash
from bon.utils.utils import load_json, save_json

LOGGER = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, cache_dir: Path, num_bins: int = 20):
        self.cache_dir = cache_dir
        self.num_bins = num_bins  # Number of bins for cache division
        self.in_memory_cache = {}

    @staticmethod
    def get_bin_number(hash_value, num_bins):
        # Convert the hash into an integer and find its modulo with num_bins
        return int(hash_value, 16) % num_bins

    def get_cache_file(self, prompt: Prompt, params: LLMParams) -> tuple[Path, str]:
        # Use the SHA-1 hash of the prompt for the dictionary key
        prompt_hash = prompt.model_hash()  # Assuming this gives a SHA-1 hash as a hex string
        bin_number = self.get_bin_number(prompt_hash, self.num_bins)

        # Construct the file name using the bin number
        cache_dir = self.cache_dir / params.model_hash()
        cache_file = cache_dir / f"bin{str(bin_number)}.json"

        return cache_file, prompt_hash

    def maybe_load_cache(self, prompt: Prompt, params: LLMParams):
        cache_file, prompt_hash = self.get_cache_file(prompt, params)
        if not cache_file.exists():
            return None

        if (cache_file not in self.in_memory_cache) or (prompt_hash not in self.in_memory_cache[cache_file]):
            LOGGER.info(f"Cache miss, loading from disk: {cache_file=}, {prompt_hash=}")
            with filelock.FileLock(str(cache_file) + ".lock"):
                self.in_memory_cache[cache_file] = load_json(cache_file)

        data = self.in_memory_cache[cache_file].get(prompt_hash, None)
        return None if data is None else LLMCache.model_validate_json(data)

    def process_cached_responses(
        self,
        prompt: Union[BatchPrompt, Prompt],
        params: LLMParams,
        n: int,
        insufficient_valids_behaviour: str,
        print_prompt_and_response: bool,
        empty_completion_threshold: float = 0.5,  # If we have multiple responses in a cache, threshold proportion of them that can be empty before we run retries
    ) -> Tuple[
        List[LLMResponse | List[LLMResponse] | None],
        List[[LLMCache | None]],
        [List[LLMResponse | List[LLMResponse] | None]],
    ]:
        """
        Process cached responses for a given prompt or batch of prompts.

        This function retrieves and processes cached results for a single prompt or a batch of prompts.
        It checks for empty completions and handles them based on a specified threshold.

        Args:
            prompt (Union[BatchPrompt, Prompt]): The prompt or batch of prompts to process.
            params (LLMParams): Parameters for the language model.
            n (int): The number of expected responses per prompt.
            insufficient_valids_behaviour (str): Behavior when there are insufficient valid responses.
            print_prompt_and_response (bool): Whether to print the prompt and its response.
            empty_completion_threshold (float, optional): The maximum allowed proportion of empty
                completions before considering the cache result as failed. Defaults to 0.5.

        Returns:
            Tuple[List[LLMResponse | List[LLMResponse]], List[[LLMCache | None]], bool]:
                - A list of LLMResponses or lists of LLMResponses for each prompt. We expect a list of lists if n > 0.
                We also only are expecting a list of responses that are in the cache. If nothing is in the cache,
                we expect an empty list
                - A list of LLMCache objects or None for each prompt.
                - A list of LLMResponse objects or None for each prompt. None is returned if there are no LLMCache failures to handle.

        Raises:
            AssertionError: If the number of cached responses is inconsistent with the expected number,
                            or if the number of responses doesn't match the number of prompts.

        Note:
            - For BatchPrompts, the function processes each prompt individually.
            - Empty completions are detected and logged, potentially marking the cache as failed.
            - The function ensures that the number of cached results matches the number of prompts.
        """
        cached_responses = []
        cached_results = []
        failed_cache_responses = []

        prompts = prompt.prompts if isinstance(prompt, BatchPrompt) else [prompt]

        for individual_prompt in prompts:
            cached_result = self.maybe_load_cache(prompt=individual_prompt, params=params)

            if cached_result is not None:
                cache_file, _ = self.get_cache_file(prompt=individual_prompt, params=params)
                LOGGER.info(f"Loaded cache for prompt from {cache_file}")

                prop_empty_completions = sum(
                    1 for response in cached_result.responses if response.completion == ""
                ) / len(cached_result.responses)

                if prop_empty_completions > empty_completion_threshold:
                    if len(cached_result.responses) == 1:
                        LOGGER.warning("Cache does not contain completion; likely due to recitation")
                    else:
                        LOGGER.warning(
                            f"Proportion of cache responses that contain empty completions ({prop_empty_completions}) is greater than threshold {empty_completion_threshold}. Likely due to recitation"
                        )
                    failed_cache_response = cached_result.responses
                    cached_result = None
                    cached_response = None
                else:
                    cached_response = (
                        cached_result.responses
                    )  # We want a list of LLMResponses if we have n responses in a cache
                    if insufficient_valids_behaviour != "continue":
                        assert (
                            len(cached_result.responses) == n
                        ), f"cache is inconsistent with n={n}\n{cached_result.responses}"
                    if print_prompt_and_response:
                        individual_prompt.pretty_print(cached_result.responses)

                    failed_cache_response = None
            else:
                cached_response = None
                cached_result = None
                failed_cache_response = None
            cached_responses.append(cached_response)
            cached_results.append(cached_result)
            failed_cache_responses.append(failed_cache_response)

        assert (
            len(cached_results) == len(prompts) == len(cached_responses) == len(failed_cache_responses)
        ), f"""Different number of cached_results, prompts, cached_responses and failed_cached_responses when they should all be length {len(prompts)}!
                Number of prompts: {len(prompts)} \n Number of cached_results: {len(cached_results)} \n Number of cached_responses: {len(cached_responses)} \n Number of failed_cache_responses: {len(failed_cache_responses)}"""

        # Check that responses is an empty list if there are no cached responses
        if all(result is None for result in cached_results):
            assert all(
                response is None for response in cached_responses
            ), f"No cached results found so responses should be length 0. Instead responses is length {len(cached_responses)}"
        return cached_responses, cached_results, failed_cache_responses

    def update_failed_cache(
        self,
        prompt: Prompt,
        responses: list[LLMResponse],
        failed_cache_responses: List[
            List[LLMResponse]
        ],  # List of previous responses from the cache that should have no completion. We use this to update api_failures
    ) -> list[LLMResponse]:
        assert len(responses) == len(
            failed_cache_responses[0]
        ), f"There should be the same number of responses and failed_cache_responses! Instead we have {len(responses)} responses and {len(failed_cache_responses)} failed_cache_responses."
        for i in range(len(responses)):
            responses[i].api_failures = failed_cache_responses[0][i].api_failures + 1

        LOGGER.info(
            f"""Updating previous failures for: \n
                    {prompt} \n
                    with responses: \n
                    {responses}"""
        )
        return responses

    def save_cache(self, prompt: Prompt, params: LLMParams, responses: list):
        cache_file, prompt_hash = self.get_cache_file(prompt, params)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        new_cache_entry = LLMCache(prompt=prompt, params=params, responses=responses)

        with filelock.FileLock(str(cache_file) + ".lock"):
            cache_data = {}
            # If the cache file exists, load it; otherwise, start with an empty dict
            if cache_file.exists():
                cache_data = load_json(cache_file)

            cache_data[prompt_hash] = new_cache_entry.model_dump_json()
            save_json(cache_file, cache_data)

    def get_moderation_file(self, texts: list[str]) -> tuple[Path, str]:
        hashes = [deterministic_hash(t) for t in texts]
        hash = deterministic_hash(" ".join(hashes))
        bin_number = self.get_bin_number(hash, self.num_bins)
        if (self.cache_dir / "moderation" / f"{hash}.json").exists():
            cache_file = self.cache_dir / "moderation" / f"{hash}.json"
        else:
            cache_file = self.cache_dir / "moderation" / f"bin{str(bin_number)}.json"

        return cache_file, hash

    def maybe_load_moderation(self, texts: list[str]):
        cache_file, hash = self.get_moderation_file(texts)
        if cache_file.exists():
            all_data = load_json(cache_file)
            data = all_data.get(hash, None)
            if data is not None:
                return LLMCacheModeration.model_validate_json(data)
        return None

    def save_moderation(self, texts: list[str], moderation: list[TaggedModeration]):
        cache_file, hash = self.get_moderation_file(texts)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        new_cache_entry = LLMCacheModeration(texts=texts, moderation=moderation)

        with filelock.FileLock(str(cache_file) + ".lock"):
            cache_data = {}
            # If the cache file exists, load it; otherwise, start with an empty dict
            if cache_file.exists():
                cache_data = load_json(cache_file)

            # Update the cache data with the new responses for this prompt
            cache_data[hash] = new_cache_entry.model_dump_json()

            save_json(cache_file, cache_data)

    def get_embeddings_file(self, params: EmbeddingParams) -> tuple[Path, str]:
        hash = params.model_hash()
        bin_number = self.get_bin_number(hash, self.num_bins)

        cache_file = self.cache_dir / "embeddings" / f"bin{str(bin_number)}.json"

        return cache_file, hash

    def maybe_load_embeddings(self, params: EmbeddingParams) -> EmbeddingResponseBase64 | None:
        cache_file, hash = self.get_embeddings_file(params)
        if cache_file.exists():
            all_data = load_json(cache_file)
            data = all_data.get(hash, None)
            if data is not None:
                return EmbeddingResponseBase64.model_validate_json(data)
        return None

    def save_embeddings(self, params: EmbeddingParams, response: EmbeddingResponseBase64):
        cache_file, hash = self.get_embeddings_file(params)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(str(cache_file) + ".lock"):
            cache_data = {}
            # If the cache file exists, load it; otherwise, start with an empty dict
            if cache_file.exists():
                cache_data = load_json(cache_file)

            # Update the cache data with the new responses for this prompt
            cache_data[hash] = response.model_dump_json()

            save_json(cache_file, cache_data)
