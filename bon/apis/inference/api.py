from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Awaitable, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from typing_extensions import Self

from bon.apis.inference.openai.utils import get_equivalent_model_ids
from bon.data_models import (
    GEMINI_MODELS,
    BatchPrompt,
    ChatMessage,
    EmbeddingParams,
    EmbeddingResponseBase64,
    LLMParams,
    LLMResponse,
    MessageRole,
    Prompt,
    TaggedModeration,
)
from bon.utils.utils import get_repo_root, load_secrets, setup_environment

from .anthropic import ANTHROPIC_MODELS, AnthropicChatModel
from .cache_manager import CacheManager
from .gemini.genai import GeminiModel
from .gemini.vertexai import GeminiVertexAIModel
from .gray_swan import GRAYSWAN_MODELS, GraySwanChatModel
from .huggingface import HUGGINGFACE_MODELS, HuggingFaceModel
from .model import InferenceAPIModel
from .openai.chat import OpenAIChatModel
from .openai.completion import OpenAICompletionModel
from .openai.embedding import OpenAIEmbeddingModel
from .openai.moderation import OpenAIModerationModel
from .openai.s2s import OpenAIS2SModel, S2SRateLimiter
from .openai.utils import COMPLETION_MODELS, GPT_CHAT_MODELS, S2S_MODELS
from .opensource.batch_inference import BATCHED_MODELS, BatchAudioModel

LOGGER = logging.getLogger(__name__)


_DEFAULT_GLOBAL_INFERENCE_API: InferenceAPI | None = None


class InferenceAPI:
    """
    A wrapper around the OpenAI and Anthropic APIs that automatically manages
    rate limits and valid responses.
    """

    def __init__(
        self,
        anthropic_num_threads: int = 80,
        openai_fraction_rate_limit: float = 0.8,
        openai_num_threads: int = 100,
        openai_s2s_num_threads: int = 40,
        gpt4o_s2s_rpm_cap: int = 10,
        gemini_num_threads: int = 120,
        gemini_recitation_rate_check_volume: int = 100,
        gemini_recitation_rate_threshold: float = 0.5,
        gray_swan_num_threads: int = 80,
        huggingface_num_threads: int = 100,
        prompt_history_dir: Path | Literal["default"] | None = "default",
        cache_dir: Path | Literal["default"] | None = "default",
        empty_completion_threshold: int = 0,
    ):
        """
        Set prompt_history_dir to None to disable saving prompt history.
        Set prompt_history_dir to "default" to use the default prompt history directory.
        """

        if openai_fraction_rate_limit > 1:
            raise ValueError("openai_fraction_rate_limit must be 1 or less")

        self.anthropic_num_threads = anthropic_num_threads
        self.openai_fraction_rate_limit = openai_fraction_rate_limit
        # limit openai api calls to stop async jamming
        self.openai_semaphore = asyncio.Semaphore(openai_num_threads)
        self.gemini_semaphore = asyncio.Semaphore(gemini_num_threads)
        self.openai_s2s_semaphore = asyncio.Semaphore(openai_s2s_num_threads)
        self.gemini_recitation_rate_check_volume = gemini_recitation_rate_check_volume
        self.gemini_recitation_rate_threshold = gemini_recitation_rate_threshold
        self.gray_swan_num_threads = gray_swan_num_threads
        self.huggingface_num_threads = huggingface_num_threads
        self.empty_completion_threshold = empty_completion_threshold
        self.gpt4o_s2s_rpm_cap = gpt4o_s2s_rpm_cap
        self.init_time = time.time()
        self.current_time = time.time()
        self.n_calls = 0
        self.gpt_4o_rate_limiter = S2SRateLimiter(self.gpt4o_s2s_rpm_cap)

        secrets = load_secrets("SECRETS")
        if prompt_history_dir == "default":
            try:
                self.prompt_history_dir = Path(secrets["PROMPT_HISTORY_DIR"])
            except KeyError:
                self.prompt_history_dir = get_repo_root() / ".prompt_history"
        else:
            assert isinstance(prompt_history_dir, Path) or prompt_history_dir is None
            self.prompt_history_dir = prompt_history_dir

        if cache_dir == "default":
            try:
                self.cache_dir = Path(secrets["CACHE_DIR"])
            except KeyError:
                self.cache_dir = get_repo_root() / ".cache"
        else:
            assert isinstance(cache_dir, Path) or cache_dir is None
            self.cache_dir = cache_dir

        self.cache_manager: CacheManager | None = None
        if self.cache_dir is not None:
            self.cache_manager = CacheManager(self.cache_dir)

        self._openai_completion = OpenAICompletionModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            prompt_history_dir=self.prompt_history_dir,
        )

        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            prompt_history_dir=self.prompt_history_dir,
        )

        self._openai_moderation = OpenAIModerationModel()

        self._openai_embedding = OpenAIEmbeddingModel()
        self._openai_s2s = OpenAIS2SModel()

        self._anthropic_chat = AnthropicChatModel(
            num_threads=self.anthropic_num_threads,
            prompt_history_dir=self.prompt_history_dir,
        )

        self._huggingface = HuggingFaceModel(
            num_threads=self.huggingface_num_threads,
            prompt_history_dir=self.prompt_history_dir,
            token=secrets["HF_API_KEY"] if "HF_API_KEY" in secrets else None,
        )

        self._gray_swan = GraySwanChatModel(
            num_threads=self.gray_swan_num_threads,
            prompt_history_dir=self.prompt_history_dir,
            api_key=secrets["GRAYSWAN_API_KEY"] if "GRAYSWAN_API_KEY" in secrets else None,
        )

        self._gemini_vertex = GeminiVertexAIModel(prompt_history_dir=self.prompt_history_dir)
        self._gemini_genai = GeminiModel(
            prompt_history_dir=self.prompt_history_dir,
            recitation_rate_check_volume=self.gemini_recitation_rate_check_volume,
            recitation_rate_threshold=self.gemini_recitation_rate_threshold,
            empty_completion_threshold=self.empty_completion_threshold,
        )
        self._batch_audio_models = {}

        # Batched models require GPU and we only want to initialize them if we have a GPU available
        print(torch.cuda.is_available())
        if torch.cuda.is_available():
            for model_name in BATCHED_MODELS:
                try:
                    self._batch_audio_models[model_name] = BatchAudioModel(
                        model_name,
                        prompt_history_dir=self.prompt_history_dir,
                    )
                except Exception as e:
                    print(f"Error loading {model_name} model: {e}")
                    self._batch_audio_models[model_name] = None

        self.running_cost: float = 0
        self.model_timings = {}
        self.model_wait_times = {}

    def semaphore_method_decorator(func):
        # unused but could be useful to debug if openai jamming comes back
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            async with self.openai_semaphore:
                return await func(self, *args, **kwargs)

        return wrapper

    @classmethod
    def get_default_global_api(cls) -> Self:
        global _DEFAULT_GLOBAL_INFERENCE_API
        if _DEFAULT_GLOBAL_INFERENCE_API is None:
            _DEFAULT_GLOBAL_INFERENCE_API = cls()
        return _DEFAULT_GLOBAL_INFERENCE_API

    @classmethod
    def default_global_running_cost(cls) -> float:
        return cls.get_default_global_api().running_cost

    def select_gemini_model(self, use_vertexai: bool = False):
        if use_vertexai:
            return self._gemini_vertex
        else:
            return self._gemini_genai

    def model_id_to_class(self, model_id: str, gemini_use_vertexai: bool = False) -> InferenceAPIModel:
        if model_id in COMPLETION_MODELS:
            return self._openai_completion
        elif model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id:
            return self._openai_chat
        elif model_id in ANTHROPIC_MODELS:
            return self._anthropic_chat
        elif model_id in HUGGINGFACE_MODELS:
            return self._huggingface
        elif model_id in GRAYSWAN_MODELS:
            return self._gray_swan
        elif model_id in GEMINI_MODELS:
            return self.select_gemini_model(gemini_use_vertexai)
        elif model_id in BATCHED_MODELS:
            class_for_model = self._batch_audio_models[model_id]
            assert class_for_model is not None, f"Error loading class for {model_id}"
            return class_for_model
        elif model_id in S2S_MODELS:
            return self._openai_s2s
        raise ValueError(f"Invalid model id: {model_id}")

    async def check_rate_limit(self, wait_time=60):
        current_time = time.time()
        total_runtime = (current_time - self.init_time) / 60
        avg_rpm = self.n_calls / total_runtime

        try:
            assert (
                avg_rpm <= self.gpt4o_s2s_rpm_cap
            ), f"Average RPM {avg_rpm} is above rate limit of {self.gpt4o_s2s_rpm_cap}!"
        except Exception:
            LOGGER.warning(
                f"Average RPM {avg_rpm} with {self.n_calls} calls made in {total_runtime} minutes is above rate limit of {self.gpt4o_s2s_rpm_cap}! Sleeping for {wait_time} seconds"
            )
            await asyncio.sleep(wait_time)
        LOGGER.info(f"Average RPM is {avg_rpm} with {self.n_calls} calls made in {total_runtime} minutes")

    def filter_responses(
        self,
        candidate_responses: list[LLMResponse],
        n: int,
        is_valid: Callable[[str], bool],
        insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids"] = "error",
    ) -> list[LLMResponse]:
        # filter out invalid responses
        num_candidates = len(candidate_responses)
        valid_responses = [response for response in candidate_responses if is_valid(response.completion)]
        num_valid = len(valid_responses)
        success_rate = num_valid / num_candidates
        if success_rate < 1:
            LOGGER.info(f"`is_valid` success rate: {success_rate * 100:.2f}%")

        # return the valid responses, or pad with invalid responses if there aren't enough
        if num_valid < n:
            match insufficient_valids_behaviour:
                case "error":
                    raise RuntimeError(f"Only found {num_valid} valid responses from {num_candidates} candidates.")
                case "retry":
                    raise RuntimeError(
                        f"Only found {num_valid} valid responses from {num_candidates} candidates. n={n}"
                    )
                case "continue":
                    responses = valid_responses
                case "pad_invalids":
                    invalid_responses = [
                        response for response in candidate_responses if not is_valid(response.completion)
                    ]
                    invalids_needed = n - num_valid
                    responses = [
                        *valid_responses,
                        *invalid_responses[:invalids_needed],
                    ]
                    LOGGER.info(
                        f"Padded {num_valid} valid responses with {invalids_needed} invalid responses to get {len(responses)} total responses"
                    )
        else:
            responses = valid_responses
        return responses[:n]

    async def __call__(
        self,
        model_ids: str | tuple[str, ...],
        prompt: Prompt | BatchPrompt,
        audio_out_dir: str | Path = None,
        print_prompt_and_response: bool = False,
        max_tokens: int | None = None,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        is_valid: Callable[[str], bool] = lambda _: True,
        insufficient_valids_behaviour: Literal["error", "continue", "pad_invalids", "retry"] = "retry",
        gemini_use_vertexai: bool = False,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should be Prompt.
            audio_out_dir : The directory where any audio output from the model (relevant for speech-to-speech models) should be saved
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            num_candidates_per_completion: How many candidate completions to generate for each desired completion.
                n*num_candidates_per_completion completions will be generated, then is_valid is applied as a filter,
                then the remaining completions are returned up to a maximum of n.
            is_valid: Candiate completions are filtered with this predicate.
            insufficient_valids_behaviour: What should we do if the remaining completions after applying the is_valid
                filter is shorter than n.
                - error: raise an error
                - continue: return the valid responses, even if they are fewer than n
                - pad_invalids: pad the list with invalid responses up to n
                - retry: retry the API call until n valid responses are found
        """

        # trick to double rate limit for most recent model only
        if isinstance(model_ids, str):
            model_ids = get_equivalent_model_ids(model_ids)

        if any(model_id in S2S_MODELS for model_id in model_ids):
            assert audio_out_dir is not None, "audio_out_dir is required when using S2S model!"

        model_classes = [self.model_id_to_class(model_id, gemini_use_vertexai) for model_id in model_ids]
        if len(set(str(type(x)) for x in model_classes)) != 1:
            raise ValueError("All model ids must be of the same type.")

        # standardize max_tokens argument
        model_class = model_classes[0]

        if isinstance(model_class, AnthropicChatModel):
            max_tokens = max_tokens if max_tokens is not None else 2000
            kwargs["max_tokens"] = max_tokens
        else:
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

        num_candidates = num_candidates_per_completion * n

        llm_cache_params = LLMParams(
            model=model_ids,
            n=n,
            insufficient_valids_behaviour=insufficient_valids_behaviour,
            num_candidates_per_completion=num_candidates_per_completion,
            **kwargs,
        )
        cached_responses, cached_results, failed_cache_responses = [], [], []

        if self.cache_manager is not None:
            cached_responses, cached_results, failed_cache_responses = self.cache_manager.process_cached_responses(
                prompt=prompt,
                params=llm_cache_params,
                n=n,
                insufficient_valids_behaviour=insufficient_valids_behaviour,
                print_prompt_and_response=print_prompt_and_response,
                empty_completion_threshold=self.empty_completion_threshold,
            )

            # If checking the cache returns results results for all prompts, then we just return the responses and don't
            # need to do further processing
            if all(cached_results):
                # Assert there is only one result in cached_result if prompt is a regular Prompt (i.e. not BatchPrompt)
                if isinstance(prompt, Prompt):
                    assert (
                        len(cached_results) == len(cached_responses) == 1
                    ), "prompt is not a BatchPrompt but cached results contain more than one entry!"
                    return cached_responses[0]
                else:
                    return cached_responses

            else:
                if isinstance(prompt, BatchPrompt):
                    # If some prompts are not cached, we need to make API calls for those
                    uncached_prompts = [p for p, c in zip(prompt.prompts, cached_results) if c is None]
                    # Check that len(responses) we get from cache + len(uncached_prompts) == len(prompts)
                    assert len([i for i in cached_responses if i is not None]) + len(uncached_prompts) == len(
                        prompt.prompts
                    ), f"""Length of responses + uncached_prompts should equal length of full batched prompts!
                        Instead there are {len(cached_responses)} responses, {len(uncached_prompts)} uncached_prompts and {len(prompt.prompts)}!"""

                    prompt = BatchPrompt(prompts=uncached_prompts)

                else:
                    # If prompt is a single prompt and there is no cached result, simply return the original prompt for regular processing
                    prompt = prompt

        if isinstance(model_class, AnthropicChatModel) or isinstance(model_class, HuggingFaceModel):
            # Anthropic chat doesn't support generating multiple candidates at once, so we have to do it manually
            candidate_responses = list(
                chain.from_iterable(
                    await asyncio.gather(
                        *[
                            model_class(
                                model_ids,
                                prompt,
                                print_prompt_and_response,
                                max_attempts_per_api_call,
                                is_valid=(is_valid if insufficient_valids_behaviour == "retry" else lambda _: True),
                                **kwargs,
                            )
                            for _ in range(num_candidates)
                        ]
                    )
                )
            )
        elif isinstance(model_class, GeminiModel) or isinstance(model_class, GeminiVertexAIModel):
            candidate_responses = []

            for _ in range(num_candidates):
                async with self.gemini_semaphore:
                    response = await model_class(
                        model_ids,
                        prompt,
                        print_prompt_and_response,
                        max_attempts_per_api_call,
                        is_valid=(
                            lambda x: (
                                x.completion != "" if insufficient_valids_behaviour == "retry" else lambda _: True
                            )
                        ),
                        **kwargs,
                    )
                    candidate_responses.extend(response)
        elif isinstance(model_class, BatchAudioModel):
            if not isinstance(prompt, BatchPrompt):
                raise ValueError(f"{model_class.__class__.__name__} requires a BatchPrompt input")
            candidate_responses = model_class(
                model_ids=model_ids,
                batch_prompt=prompt,
                print_prompt_and_response=print_prompt_and_response,
                max_attempts_per_api_call=max_attempts_per_api_call,
                n=num_candidates,
                is_valid=(is_valid if insufficient_valids_behaviour == "retry" else lambda _: True),
                **kwargs,
            )

            # Update candidate_responses to include responses from cache
            candidate_iter = iter(candidate_responses)
            if self.cache_manager is not None:
                candidate_responses = [
                    next(candidate_iter) if response is None else response for response in cached_responses
                ]
            else:
                candidate_responses = candidate_responses

        elif isinstance(model_class, OpenAIS2SModel):
            candidate_responses = []
            for _ in range(num_candidates):
                async with self.openai_s2s_semaphore:
                    self.n_calls += 1
                    await self.gpt_4o_rate_limiter.acquire()
                    response = await model_class(
                        model_ids=model_ids,
                        prompt=prompt,
                        audio_out_dir=audio_out_dir,
                        print_prompt_and_response=print_prompt_and_response,
                        max_attempts_per_api_call=max_attempts_per_api_call,
                        is_valid=(is_valid if insufficient_valids_behaviour == "retry" else lambda _: True),
                        **kwargs,
                    )
                    candidate_responses.extend(response)
        else:
            async with self.openai_semaphore:
                candidate_responses = await model_class(
                    model_ids,
                    prompt,
                    print_prompt_and_response,
                    max_attempts_per_api_call,
                    n=num_candidates,
                    is_valid=(is_valid if insufficient_valids_behaviour == "retry" else lambda _: True),
                    **kwargs,
                )

        # Save cache
        if self.cache_manager is not None:
            # Handling for empty cache due to recitation errors for Gemini
            if any(failed_cache_responses):
                assert isinstance(prompt, Prompt), "Models that use BatchPrompt should not have a failed_cache_response"
                candidate_responses = self.cache_manager.update_failed_cache(
                    prompt=prompt, responses=candidate_responses, failed_cache_responses=failed_cache_responses
                )

            # Handle cache saving and output for batch_prompts
            if isinstance(prompt, BatchPrompt):
                for i, (p, response) in enumerate(zip(prompt.prompts, candidate_responses)):
                    self.cache_manager.save_cache(
                        prompt=p,
                        params=llm_cache_params,
                        responses=[response],
                    )

            else:
                self.cache_manager.save_cache(
                    prompt=prompt,
                    params=llm_cache_params,
                    responses=candidate_responses,
                )
        # filter based on is_valid criteria and insufficient_valids_behaviour
        # Only do this for non-batched inputs
        if isinstance(prompt, Prompt):
            responses = self.filter_responses(
                candidate_responses,
                n,
                is_valid,
                insufficient_valids_behaviour,
            )
        else:
            assert (
                insufficient_valids_behaviour == "retry"
            ), f"We don't support {insufficient_valids_behaviour} for BatchPrompt"
            responses = candidate_responses

        # update running cost and model timings
        self.running_cost += sum(response.cost for response in candidate_responses)
        for response in candidate_responses:
            self.model_timings.setdefault(response.model_id, []).append(response.api_duration)
            self.model_wait_times.setdefault(response.model_id, []).append(response.duration - response.api_duration)

        return responses

    async def ask_single_question(
        self,
        model_ids: str | tuple[str, ...],
        question: str,
        system_prompt: str | None = None,
        **api_kwargs,
    ) -> list[str]:
        """Wrapper around __call__ to ask a single question."""
        responses = await self(
            model_ids=model_ids,
            prompt=Prompt(
                messages=(
                    [] if system_prompt is None else [ChatMessage(role=MessageRole.system, content=system_prompt)]
                )
                + [
                    ChatMessage(role=MessageRole.user, content=question),
                ]
            ),
            **api_kwargs,
        )

        return [r.completion for r in responses]

    async def moderate(
        self,
        texts: list[str],
        model_id: str = "text-moderation-latest",
        progress_bar: bool = False,
    ) -> list[TaggedModeration] | None:
        """
        Returns moderation results on text. Free to use with no rate limit, but
        should only be called with outputs from OpenAI models.
        """
        if self.cache_manager is not None:
            cached_result = self.cache_manager.maybe_load_moderation(texts)
            if cached_result is not None:
                return cached_result.moderation

        moderation_result = await self._openai_moderation(
            model_id=model_id,
            texts=texts,
            progress_bar=progress_bar,
        )
        if self.cache_manager is not None:
            self.cache_manager.save_moderation(texts, moderation_result)

        return moderation_result

    async def _embed_single_batch(
        self,
        params: EmbeddingParams,
    ):
        if self.cache_manager is not None:
            cached_result = self.cache_manager.maybe_load_embeddings(params)
            if cached_result is not None:
                return cached_result

        response = await self._openai_embedding.embed(params)

        if self.cache_manager is not None:
            self.cache_manager.save_embeddings(params, response)

        self.running_cost += response.cost
        return response

    async def embed(
        self,
        texts: list[str],
        model_id: str = "text-embedding-3-large",
        dimensions: int | None = None,
        progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Returns embeddings for text.
        TODO: Actually implement a proper rate limit.
        """
        request_futures: list[Awaitable[EmbeddingResponseBase64]] = []

        batch_size = self._openai_embedding.batch_size
        for beg_idx in range(0, len(texts), batch_size):
            batch_texts = texts[beg_idx : beg_idx + batch_size]
            request_futures.append(
                self._embed_single_batch(
                    params=EmbeddingParams(
                        model_id=model_id,
                        texts=batch_texts,
                        dimensions=dimensions,
                    )
                )
            )

        async_lib = tqdm if progress_bar else asyncio
        responses = await async_lib.gather(*request_futures)

        return np.concatenate([r.get_numpy_embeddings() for r in responses])

    def reset_cost(self):
        self.running_cost = 0

    def log_model_timings(self):
        if len(self.model_timings) > 0:
            plt.figure(figsize=(10, 6))
            for model in self.model_timings:
                timings = np.array(self.model_timings[model])
                wait_times = np.array(self.model_wait_times[model])
                LOGGER.info(
                    f"{model}: response {timings.mean():.3f}, waiting {wait_times.mean():.3f} (max {wait_times.max():.3f}, min {wait_times.min():.3f})"
                )
                plt.plot(
                    timings,
                    label=f"{model} - Response Time",
                    linestyle="-",
                    linewidth=2,
                )
                plt.plot(
                    wait_times,
                    label=f"{model} - Waiting Time",
                    linestyle="--",
                    linewidth=2,
                )
            plt.legend()
            plt.title("Model Performance: Response and Waiting Times")
            plt.xlabel("Sample Number")
            plt.ylabel("Time (seconds)")
            plt.savefig(
                self.prompt_history_dir / "model_timings.png",
                bbox_inches="tight",
            )
            plt.close()


async def demo():
    setup_environment()
    model_api = InferenceAPI()

    prompt_examples = [
        [
            {"role": "system", "content": "You are a comedic pirate."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {
                "role": "system",
                "content": "You are a swashbuckling space-faring voyager.",
            },
            {"role": "user", "content": "Hello!"},
        ],
        [
            {
                "role": "system",
                "content": "You are a swashbuckling space-faring voyager.",
            },
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How dare you call me a"},
        ],
        [
            {"role": "none", "content": "whence afterever the storm"},
        ],
    ]
    prompts = [Prompt(messages=messages) for messages in prompt_examples]

    # test anthropic chat, and continuing assistant message
    anthropic_requests = [
        model_api(
            "claude-3-opus-20240229",
            prompt=prompts[0],
            n=1,
            print_prompt_and_response=True,
        ),
        model_api(
            "claude-3-opus-20240229",
            prompt=prompts[2],
            n=1,
            print_prompt_and_response=True,
        ),
    ]

    # test OAI chat, more than 1 model and n > 1
    oai_chat_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]
    oai_chat_requests = [
        model_api(
            oai_chat_models,
            prompt=prompts[1],
            n=6,
            print_prompt_and_response=True,
        ),
    ]

    # test OAI completion with none message and assistant message to continue
    oai_requests = [
        model_api(
            "gpt-3.5-turbo-instruct",
            prompt=prompts[2],
            print_prompt_and_response=True,
        ),
        model_api(
            "gpt-3.5-turbo-instruct",
            prompt=prompts[3],
            print_prompt_and_response=True,
        ),
    ]
    answer = await asyncio.gather(*anthropic_requests, *oai_chat_requests, *oai_requests)

    costs = defaultdict(int)
    for responses in answer:
        for response in responses:
            costs[response.model_id] += response.cost

    print("-" * 80)
    print("Costs:")
    for model_id, cost in costs.items():
        print(f"{model_id}: ${cost}")
    return answer


if __name__ == "__main__":
    asyncio.run(demo())
