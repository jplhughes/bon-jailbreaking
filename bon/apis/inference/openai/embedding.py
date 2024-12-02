import asyncio
import logging
import time
import traceback

import openai
import openai._models
import openai.types

from bon.data_models import EmbeddingParams, EmbeddingResponseBase64

from .utils import EMBEDDING_MODELS, price_per_token

LOGGER = logging.getLogger(__name__)


class OpenAIEmbeddingModel:
    def __init__(self):
        # TODO: Actually implement a proper rate limit
        self.num_threads = 1
        self.batch_size = 2048  # Max batch size for embedding endpoint

        self.aclient = openai.AsyncClient()
        self.available_requests = asyncio.BoundedSemaphore(self.num_threads)

    async def embed(
        self,
        params: EmbeddingParams,
        max_attempts: int = 5,
        **kwargs,
    ) -> EmbeddingResponseBase64:
        assert params.model_id in EMBEDDING_MODELS

        if params.dimensions is not None:
            kwargs["dimensions"] = params.dimensions
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    # We use base64 format.
                    # See for details https://github.com/EliahKagan/embed-encode/blob/main/why.md
                    response = await self.aclient.embeddings.create(
                        model=params.model_id,
                        input=params.texts,
                        encoding_format="base64",
                        **kwargs,
                    )

                assert all(isinstance(x.embedding, str) for x in response.data)
                total_tokens = response.usage.total_tokens

                return EmbeddingResponseBase64(
                    model_id=response.model,
                    embeddings=[str(x.embedding) for x in response.data],
                    tokens=total_tokens,
                    cost=price_per_token(response.model)[0] * total_tokens,
                )

            except Exception as e:
                error_info = (
                    f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {traceback.format_exc()}"
                )
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                time.sleep(1.5**i)

        raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")
