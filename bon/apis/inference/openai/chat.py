import collections
import logging
import os
import time

import openai
import openai.types
import openai.types.chat
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from bon.data_models import LLMResponse, Prompt
from bon.utils import math_utils

from .base import OpenAIModel
from .utils import GPT_CHAT_MODELS, VISION_MODELS, get_rate_limit, price_per_token

LOGGER = logging.getLogger(__name__)


class OpenAIChatModel(OpenAIModel):
    def _assert_valid_id(self, model_id: str):
        if "ft:" in model_id:
            model_id = model_id.split(":")[1]
        assert model_id in GPT_CHAT_MODELS, f"Invalid model id: {model_id}"

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/chat/completions"
        api_key = os.environ["OPENAI_API_KEY"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 1"}],
        }
        response = requests.post(url, headers=headers, json=data)
        if "x-ratelimit-limit-tokens" not in response.headers:
            tpm, rpm = get_rate_limit(model_id)
            print(f"Failed to get dummy response header {model_id}, setting to tpm, rpm: {tpm}, {rpm}")
            header_dict = dict(response.headers)
            return header_dict | {
                "x-ratelimit-limit-tokens": str(tpm),
                "x-ratelimit-limit-requests": str(rpm),
                "x-ratelimit-remaining-tokens": str(tpm),
                "x-ratelimit-remaining-requests": str(rpm),
            }
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: Prompt, **kwargs) -> int:
        # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
        BUFFER = 5  # A bit of buffer for some error margin
        MIN_NUM_TOKENS = 20

        num_tokens = 0
        for message in prompt.messages:
            num_tokens += 1
            num_tokens += len(message.content) / 4

        return max(
            MIN_NUM_TOKENS,
            int(num_tokens + BUFFER) + kwargs.get("n", 1) * kwargs.get("max_tokens", 15),
        )

    @staticmethod
    def convert_top_logprobs(data):
        # convert OpenAI chat version of logprobs response to completion version
        top_logprobs = []

        for item in data.content:
            # <|end|> is known to be duplicated on gpt-4-turbo-2024-04-09
            # See experiments/sample-notebooks/api-tests/logprob-dup-tokens.ipynb for details
            possibly_duplicated_top_logprobs = collections.defaultdict(list)
            for top_logprob in item.top_logprobs:
                possibly_duplicated_top_logprobs[top_logprob.token].append(top_logprob.logprob)

            top_logprobs.append({k: math_utils.logsumexp(vs) for k, vs in possibly_duplicated_top_logprobs.items()})

        return top_logprobs

    async def _make_api_call(self, prompt: Prompt, model_id, start_time, **params) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call")

        if prompt.contains_image():
            assert model_id in VISION_MODELS, f"Model {model_id} does not support images"

        # convert completion logprobs api to chat logprobs api
        if "logprobs" in params:
            params["top_logprobs"] = params["logprobs"]
            params["logprobs"] = True

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)
        api_start = time.time()
        api_response: openai.types.chat.ChatCompletion = await self.aclient.chat.completions.create(
            messages=prompt.openai_format(),
            model=model_id,
            **params,
        )
        api_duration = time.time() - api_start
        duration = time.time() - start_time
        context_token_cost, completion_token_cost = price_per_token(model_id)
        context_cost = api_response.usage.prompt_tokens * context_token_cost
        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice.message.content,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=context_cost + self.count_tokens(choice.message.content) * completion_token_cost,
                logprobs=(self.convert_top_logprobs(choice.logprobs) if choice.logprobs is not None else None),
            )
            for choice in api_response.choices
        ]
        self.add_response_to_prompt_file(prompt_file, responses)
        return responses

    async def make_stream_api_call(
        self,
        prompt: Prompt,
        model_id,
        **params,
    ) -> openai.AsyncStream[openai.types.chat.ChatCompletionChunk]:
        # TODO(tony): Can eventually wrap this in an async generator and keep
        #             track of cost. Will need to do this in the parent API
        #             class.
        return await self.aclient.chat.completions.create(
            messages=prompt.openai_format(),
            model=model_id,
            stream=True,
            **params,
        )
