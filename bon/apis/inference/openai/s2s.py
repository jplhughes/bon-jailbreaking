import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import websockets
from pydub import AudioSegment
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from websockets.exceptions import WebSocketException

from bon.data_models import LLMResponse, Prompt
from bon.data_models.hashable import deterministic_hash

from ..model import InferenceAPIModel

LOGGER = logging.getLogger(__name__)


class S2SRateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.tokens = calls_per_minute
        self.last_refill_time = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_refill_time
            self.tokens = min(self.calls_per_minute, self.tokens + time_passed * (self.calls_per_minute / 60))
            self.last_refill_time = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.calls_per_minute / 60)
                LOGGER.info(f"Waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1


class OpenAIS2SModel(InferenceAPIModel):
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.model = "gpt-4o-realtime-preview-2024-10-01"
        self.max_size = 10 * 1024 * 1024  # 10MB
        self.base_wait = 10
        self.max_wait = 60
        self.max_attempts = 10

        self.allowed_kwargs = {"temperature", "max_output_tokens", "voice", "audio_format"}

    def log_retry(self, retry_state):
        if retry_state.attempt_number > 1:
            exception = retry_state.outcome.exception()
            LOGGER.warning(f"Retry attempt {retry_state.attempt_number} after error: {str(exception)}")

    def process_responses(self, audio_output: List[bytes], text_response: str, audio_out_dir: Path | str):
        audio_filename = audio_out_dir / Path(deterministic_hash(text_response) + ".wav")

        combined_audio = AudioSegment.empty()
        for audio_bytes in audio_output:
            segment = AudioSegment(data=base64.b64decode(audio_bytes), sample_width=2, frame_rate=24000, channels=1)
            combined_audio += segment

        combined_audio.export(audio_filename, format="wav")
        LOGGER.info(f"Successfully saved response data with {len(audio_output)} chunks to {audio_filename}")
        return audio_filename

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=10, max=60),
        retry=retry_if_exception_type((TimeoutError, WebSocketException, asyncio.exceptions.CancelledError)),
        after=log_retry,
    )
    async def connect(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "realtime=v1",
        }
        LOGGER.info("CONNECTION REQUEST STARTED")
        return await websockets.connect(
            f"{self.base_url}?model={self.model}", extra_headers=headers, max_size=self.max_size
        )

    async def send_message(self, websocket: websockets.WebSocketClientProtocol, message: Dict[str, Any]) -> None:
        await websocket.send(json.dumps(message))

    async def receive_message(self, websocket: websockets.WebSocketClientProtocol) -> Dict[str, Any]:
        response = await websocket.recv()
        return json.loads(response)

    async def send_text_input(self, websocket: websockets.WebSocketClientProtocol, text_input_data: List[str]) -> None:
        for text_input in text_input_data:
            text_event = {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text_input}]},
            }
            await self.send_message(websocket, text_event)

    async def send_audio_input(
        self, websocket: websockets.WebSocketClientProtocol, audio_input_data: List[str]
    ) -> None:
        for audio_chunk in audio_input_data:
            audio_event = {"type": "input_audio_buffer.append", "audio": audio_chunk}
            await self.send_message(websocket, audio_event)

        commit_event = {"type": "input_audio_buffer.commit"}
        await self.send_message(websocket, commit_event)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=10, max=60),
        retry=retry_if_exception_type((TimeoutError, WebSocketException, asyncio.exceptions.CancelledError)),
        after=log_retry,
    )
    async def run_query(
        self, input_data: List[bytes] | List[str], audio_out_dir: str | Path, params: Dict[str, Any]
    ) -> List[Any]:
        websocket = await self.connect()
        try:
            # Handle session creation
            session_created = await self.receive_message(websocket)
            LOGGER.info(f"Session created: {session_created}")

            # Send session configuration
            config = {
                "type": "session.update",
                "session": {
                    "turn_detection": {"type": "server_vad", "threshold": 0.0001},
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    **params,
                },
            }
            await self.send_message(websocket, config)

            # Detect whether we have audio or text data
            assert len(input_data) > 0, "No input data detected!"

            # Check to see if input_data is base64 encoded text (audio input) or just regular text (we use a simple check looking at the number of /)

            if input_data[0].count("/") < 10:
                await self.send_text_input(websocket, input_data)

            # Send audio data
            else:
                await self.send_audio_input(websocket, input_data)

            # Signal end of client turn
            await self.send_message(websocket, {"type": "response.create"})

            # Process responses
            audio_responses = []
            done_received = False
            last_event_time = time.time()
            timeout = 5.0  # Set the timeout in seconds
            while True:
                try:
                    response = await asyncio.wait_for(self.receive_message(websocket), timeout=timeout)
                    last_event_time = time.time()

                    if response["type"] == "response.done":
                        LOGGER.info(response)
                        transcript = response["response"]["output"][0]["content"][0]["transcript"]
                        text_out = transcript
                        done_received = True
                    elif response["type"] == "response.audio.delta":
                        audio_responses.append(response["delta"])
                    elif response["type"] != "response.audio_transcript.delta":
                        LOGGER.info(response)
                except asyncio.TimeoutError:
                    if done_received and (time.time() - last_event_time) >= timeout:
                        LOGGER.info(f"No events received for {timeout} seconds after 'response.done'. Ending loop.")
                        break

            audio_out_file = self.process_responses(audio_responses, text_out, audio_out_dir)
            LOGGER.info(text_out)
            return audio_out_file, text_out
        finally:
            await websocket.close()

    async def __call__(
        self,
        model_ids: str | tuple[str, ...],
        prompt: Prompt,
        audio_out_dir: str | Path,
        print_prompt_and_response: bool = False,
        max_attempts: int = 10,
        **kwargs,
    ) -> List[LLMResponse]:

        self.max_attempts = max_attempts

        assert len(model_ids) == 1, "Implementation only supports one model at a time."

        (model_id,) = model_ids

        start = time.time()
        input_data = prompt.openai_s2s_format()
        assert isinstance(input_data, List), "Input data format must be a list!"
        params = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
        try:
            api_start = time.time()
            audio_out_file, text_out = await self.run_query(
                input_data=input_data, audio_out_dir=audio_out_dir, params=params
            )
            api_duration = time.time() - api_start
        except Exception as e:
            LOGGER.error(f"Failed to complete query after {self.max_attempts} attempts: {str(e)}")
            return [
                LLMResponse(
                    model_id=model_id,
                    completion="",
                    stop_reason="api_error",
                    cost=0,
                    duration=time.time() - start,
                    api_duration=time.time() - api_start,
                )
            ]

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        response = LLMResponse(
            model_id=model_id,
            completion=text_out,
            audio_out=str(audio_out_file),
            stop_reason="max_tokens",
            duration=duration,
            api_duration=api_duration,
            cost=0,
        )
        return [response]
