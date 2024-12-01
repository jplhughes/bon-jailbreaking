import logging
import os
import time

import openai
import openai.types
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from almj.data_models import LLMResponse, Prompt

from .base import OpenAIModel
from .utils import COMPLETION_MODELS, get_rate_limit, price_per_token

LOGGER = logging.getLogger(__name__)


class OpenAICompletionModel(OpenAIModel):
    def _assert_valid_id(self, model_id: str):
        assert model_id in COMPLETION_MODELS, f"Invalid model id: {model_id}"

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/completions"
        api_key = os.environ["OPENAI_API_KEY"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {"model": model_id, "prompt": "a", "max_tokens": 1}
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

    def _count_prompt_token_capacity(self, prompt: Prompt, **kwargs) -> int:
        max_tokens = kwargs.get("max_tokens", 15)
        n = kwargs.get("n", 1)
        completion_tokens = n * max_tokens

        prompt_tokens = self.count_tokens(str(prompt))
        return prompt_tokens + completion_tokens

    async def _make_api_call(self, prompt: Prompt, model_id, start_time, **params) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call")
        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)
        api_start = time.time()
        api_response: openai.types.Completion = await self.aclient.completions.create(
            prompt=str(prompt),
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
                completion=choice.text,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=context_cost + self.count_tokens(choice.text) * completion_token_cost,
                logprobs=(choice.logprobs.top_logprobs if choice.logprobs is not None else None),
            )
            for choice in api_response.choices
        ]
        self.add_response_to_prompt_file(prompt_file, responses)
        return responses
