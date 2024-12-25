import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc

import anthropic.types
from anthropic import AsyncAnthropic

from bon.data_models import LLMResponse, Prompt

from .model import InferenceAPIModel

ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
}
VISION_MODELS = {
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
}
LOGGER = logging.getLogger(__name__)


class AnthropicChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.aclient = AsyncAnthropic()  # Assuming AsyncAnthropic has a default constructor
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))
        self.allowed_kwargs = {"temperature", "max_tokens"}

    async def __call__(
        self,
        model_ids: tuple[str, ...],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert len(model_ids) == 1, "Anthropic implementation only supports one model at a time."

        (model_id,) = model_ids

        if prompt.contains_image():
            assert model_id in VISION_MODELS, f"Model {model_id} does not support images"

        sys_prompt, chat_messages = prompt.anthropic_format()
        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response: anthropic.types.Message | None = None
        duration = None

        error_list = []
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
                    response = await self.aclient.messages.create(
                        messages=chat_messages,
                        model=model_id,
                        **filtered_kwargs,
                        **(dict(system=sys_prompt) if sys_prompt else {}),
                    )

                    api_duration = time.time() - api_start
                    if not is_valid(response):
                        raise RuntimeError(f"Invalid response according to is_valid {response}")
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                error_list.append(error_info)
                api_duration = time.time() - api_start
                await asyncio.sleep(1.5**i)
            else:
                break

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")
        if response is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        else:
            assert len(response.content) == 1  # Anthropic doesn't support multiple completions (as of 2024-03-12).

            response = LLMResponse(
                model_id=model_id,
                completion=response.content[0].text,
                stop_reason=response.stop_reason,
                duration=duration,
                api_duration=api_duration,
                cost=0,
            )
        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        responses = [response]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses

    def make_stream_api_call(
        self,
        model_id: str,
        prompt: Prompt,
        max_tokens: int,
        **params,
    ) -> anthropic.AsyncMessageStreamManager:
        sys_prompt, chat_messages = prompt.anthropic_format()
        return self.aclient.messages.stream(
            model=model_id,
            messages=chat_messages,
            **(dict(system=sys_prompt) if sys_prompt else {}),
            max_tokens=max_tokens,
            **params,
        )
