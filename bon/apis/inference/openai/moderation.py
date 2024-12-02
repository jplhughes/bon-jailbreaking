import asyncio
import logging
import time
from traceback import format_exc
from typing import Awaitable

import openai
import openai._models
import openai.types
from tqdm.asyncio import tqdm_asyncio

from bon.data_models.inference import TaggedModeration

LOGGER = logging.getLogger(__name__)


class OpenAIModerationModel:
    def __init__(
        self,
        num_threads: int = 100,
    ):
        """
        num_threads: number of concurrent requests to make to OpenAI API.
        The moderation endpoint is supposed to have no rate limit, but we set
        a limit here to be safe and avoid DDOS preventions from kicking in.
        """
        self.num_threads = num_threads
        self._batch_size = 32  # Max batch size for moderation endpoint

        self.aclient = openai.AsyncClient()
        self.available_requests = asyncio.BoundedSemaphore(self.num_threads)

    async def _single_moderation_request(
        self,
        model_id: str,
        texts: list[str],
        max_attempts: int,
    ) -> openai.types.ModerationCreateResponse:
        assert len(texts) <= self._batch_size

        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    response = await self.aclient.moderations.create(
                        model=model_id,
                        input=texts,
                    )
                return response
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                time.sleep(1.5**i)

        raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

    async def __call__(
        self,
        model_id: str,
        texts: list[str],
        max_attempts: int = 5,
        progress_bar: bool = False,
    ) -> list[TaggedModeration]:
        """
        texts: list of texts to moderate
        """
        request_futures: list[Awaitable[openai.types.ModerationCreateResponse]] = []
        for beg_idx in range(0, len(texts), self._batch_size):
            batch_texts = texts[beg_idx : beg_idx + self._batch_size]
            request_futures.append(
                self._single_moderation_request(
                    model_id=model_id,
                    texts=batch_texts,
                    max_attempts=max_attempts,
                )
            )

        async_lib = tqdm_asyncio if progress_bar else asyncio
        responses = await async_lib.gather(*request_futures)

        moderation_results = []
        for response in responses:
            for moderation in response.results:
                moderation_results.append(
                    TaggedModeration(
                        model_id=response.model,
                        moderation=moderation,
                    )
                )

        return moderation_results
