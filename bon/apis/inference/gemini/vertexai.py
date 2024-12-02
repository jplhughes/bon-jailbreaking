import asyncio
import logging
import os
import time
from pathlib import Path
from traceback import format_exc
from typing import Callable, List, Optional

import google.generativeai as genai
import googleapiclient
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, HarmBlockThreshold, HarmCategory

from bon.data_models import GeminiStopReason, LLMResponse, Prompt

from ....data_models.utils import (
    DELETE_FILE_QUOTA,
    GEMINI_MODELS,
    GEMINI_RPM,
    GEMINI_TPM,
    GeminiRateTracker,
    async_delete_genai_files,
    get_block_reason,
    get_stop_reason,
    parse_safety_ratings,
)
from ..model import InferenceAPIModel

LOGGER = logging.getLogger(__name__)


class GeminiVertexAIModel(InferenceAPIModel):
    def __init__(
        self,
        prompt_history_dir: Path = None,
        project: str = "jailbreak-defense",
        location: str = "us-west1",
    ):
        self.project = project
        self.location = location
        self.prompt_history_dir = prompt_history_dir
        self.model_ids = set()
        self.rate_trackers = {}
        self.lock = asyncio.Lock()

        self.kwarg_change_name = {
            "temperature": "temperature",
            "max_tokens": "max_output_tokens",
            "top_p": "top_p",
            "top_k": "top_k",
        }

        # Maps simple input of the safety threshold values (None, few, some, most) to the HarmBlockThreshold class values
        self.map_safety_block_name = {
            None: HarmBlockThreshold.BLOCK_NONE,
            "few": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "some": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "most": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        vertexai.init(project=os.environ["GOOGLE_PROJECT_ID"], location=os.environ["GOOGLE_PROJECT_REGION"])

    @staticmethod
    def _print_prompt_and_response(prompt, responses):
        raise NotImplementedError

    def add_model_id(self, model_id: str):
        if model_id not in GEMINI_MODELS:
            raise ValueError(f"Unsupported model: {model_id}")
        if model_id not in self.model_ids:
            self.model_ids.add(model_id)
            self.rate_trackers[model_id] = GeminiRateTracker(rpm_limit=GEMINI_RPM[model_id], tpm_limit=GEMINI_TPM)

    def get_generation_config(self, kwargs):
        params = {self.kwarg_change_name[k]: v for k, v in kwargs.items() if k in self.kwarg_change_name}
        return GenerationConfig(**params)

    def get_safety_settings(self, safety_threshold):
        safety_settings = {
            category: self.map_safety_block_name[safety_threshold]
            for category in [
                HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                HarmCategory.HARM_CATEGORY_HARASSMENT,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        }
        return safety_settings

    async def run_query(
        self,
        model_id: str,
        prompt: Prompt,
        generation_config: GenerationConfig = None,
    ) -> str:

        if generation_config:
            model = GenerativeModel(model_name=model_id, generation_config=generation_config)
        else:
            model = GenerativeModel(model_name=model_id)

        # Get audio file to delete later
        content = prompt.gemini_format()
        audio_input_files = [i for i in content if isinstance(i, genai.types.file_types.File)]
        response = await model.generate_content_async(contents=content)
        return response, audio_input_files

    async def _make_api_call(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid: Callable[[str], bool] = lambda x: x.stop_reason != GeminiStopReason.RECITATION,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()

        api_start = time.time()
        generation_config = self.get_generation_config(kwargs)

        response_data, audio_input_files = await self.run_query(
            model_id=model_id, prompt=prompt, generation_config=generation_config
        )

        api_duration = time.time() - api_start

        if response_data is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        try:
            if not response_data.candidates:
                LOGGER.info("NO CANDIDATE RESPONSES")
                block_reason = response_data.prompt_feedback.block_reason
                LOGGER.info(f"blockreason {block_reason}")
                LOGGER.info(f"No candidates return on prompt {prompt}")
                LOGGER.info(f"Response: {response_data}")

                response = LLMResponse(
                    model_id=model_id,
                    completion="",
                    stop_reason=get_block_reason(block_reason),
                    safety_ratings={},
                    duration=duration,
                    api_duration=api_duration,
                    cost=0,
                )
            else:
                try:
                    completion = response_data.candidates[0].content.parts[0].text
                    stop_reason = response_data.candidates[0].finish_reason

                    response = LLMResponse(
                        model_id=model_id,
                        completion=completion,
                        stop_reason=get_stop_reason(stop_reason),
                        safety_ratings=parse_safety_ratings(safety_ratings=response_data.candidates[0].safety_ratings),
                        duration=duration,
                        api_duration=api_duration,
                        cost=0,
                    )
                except Exception:
                    LOGGER.info("CANDIDATE RESPONSE DOES NOT CONTAIN COMPLETION. JUST EXTRACTING SAFETY RATINGS")
                    response = LLMResponse(
                        model_id=model_id,
                        completion="",
                        stop_reason=get_stop_reason(response_data.candidates[0].finish_reason),
                        safety_ratings=parse_safety_ratings(response_data.candidates[0].safety_ratings),
                        duration=duration,
                        api_duration=api_duration,
                        cost=0,
                    )

        except Exception as e:
            LOGGER.error(f"Error parsing response: {str(e)}")
            LOGGER.error(f"Response: {response_data}")
            LOGGER.error(f"Response prompt_feedback: {response_data.prompt_feedback}")

        # Clean up and remove audio file object
        for audio_input_file in audio_input_files:
            try:
                audio_input_file.delete()
                LOGGER.info(f"Successfully deleted file {audio_input_file.name}")
            except Exception:
                LOGGER.error(f"Error deleting file {audio_input_file.name}")
        return [response]

    async def __call__(
        self,
        model_ids: tuple[str, ...],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid: Callable[[str], bool] = lambda x: x.stop_reason != GeminiStopReason.RECITATION,
        safety_threshold: str = None,
        **kwargs,
    ) -> List[LLMResponse]:
        start = time.time()

        for model_id in model_ids:
            self.add_model_id(model_id)

        async def attempt_api_call(model_id):
            async with self.lock:
                rate_tracker = self.rate_trackers[model_id]
                if rate_tracker.can_make_request(GEMINI_RPM[model_id]):
                    rate_tracker.add_request(
                        GenerativeModel(model_name=model_id)
                        .count_tokens(prompt.gemini_format(use_vertexai=True))
                        .total_tokens
                    )
                else:
                    return None

            # try:
            return await self._make_api_call(
                model_id, prompt, print_prompt_and_response, max_attempts, is_valid, **kwargs
            )
            # except Exception as e:
            #     LOGGER.warning(f"Error calling {model_id}: {str(e)}")
            #     return None

        responses: Optional[List[LLMResponse]] = None

        for i in range(max_attempts):
            try:
                responses = await attempt_api_call(model_ids[0])
                if responses is not None and not all(is_valid(response) for response in responses):
                    raise RuntimeError(f"Invalid responses according to is_valid {responses}")
            except googleapiclient.errors.ResumableUploadError as e:
                if "Resource Exhausted" in str(e):

                    # Delete 1000 oldest files current on the GenerativeAI File Storage system
                    files = [f for f in genai.list_files()]
                    LOGGER.warning(
                        f"Hit Google.GenerativeAI file system quota. There are current {len(files)} files. Attempting to delete {DELETE_FILE_QUOTA} oldest files"
                    )
                    files_to_delete = files[-DELETE_FILE_QUOTA:]
                    await async_delete_genai_files(files_to_delete)
                    files_new = [f for f in genai.list_files()]
                    print(f"Successfully deleted files. Current file system usage: {len(files_new)}")

            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break

        if responses is None:
            LOGGER.error(f"Failed to get a response for {prompt} from any API after {max_attempts} attempts.")
            return [{}]
        else:
            # Add number of attempts to response object for recitation retries
            for response in responses:
                response.recitation_retries = i

            if print_prompt_and_response:
                prompt.pretty_print(responses)

            LOGGER.debug(f"Completed call to Gemini in {time.time() - start}s.")
            return responses
