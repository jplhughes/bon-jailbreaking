import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc

from gray_swan import AsyncGraySwan

from bon.data_models import LLMResponse, Prompt

from .model import InferenceAPIModel

GRAYSWAN_MODELS = {"cygnet"}
LOGGER = logging.getLogger(__name__)


class GraySwanChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        api_key: str | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        if api_key is not None:
            self.aclient = AsyncGraySwan(api_key=api_key)
        else:
            LOGGER.warning("GRAYSWAN_API_KEY not found in secrets, GraySwanChatModel will not be available")
            self.aclient = None
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
        if self.aclient is None:
            raise RuntimeError("GraySwanChatModel is not available, please set GRAYSWAN_API_KEY in secrets")

        start = time.time()
        assert len(model_ids) == 1, "Cygnet implementation only supports one model at a time."
        (model_id,) = model_ids

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response = None
        duration = None

        error_list = []
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
                    response = await self.aclient.chat.completion.create(
                        messages=prompt.openai_format(),
                        model=model_id,
                        stream=False,
                        **filtered_kwargs,
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

        assert len(response.choices) == 1, f"Expected 1 choice, got {len(response.choices)}"

        response = LLMResponse(
            model_id=model_id,
            completion=response.choices[0].message.content,
            stop_reason=response.choices[0].finish_reason,
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
