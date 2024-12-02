import time
from pathlib import Path
from typing import Callable, List, Tuple

import torch

from bon.data_models import BatchPrompt, LLMResponse

from ..model import InferenceAPIModel

BATCHED_MODELS = {"DiVA": "meta-llama/Meta-Llama-3-8B-Instruct"}


class BatchAudioModel(InferenceAPIModel):
    def __new__(cls, model_name: str, prompt_history_dir: Path = None):
        if cls is BatchAudioModel:
            if model_name == "DiVA":
                return super().__new__(DivaModel)
            # Add more subclasses here as needed
        return super().__new__(cls)

    def __init__(self, model_name: str, prompt_history_dir: Path = None):
        self.model_name = model_name
        self.prompt_history_dir = prompt_history_dir

    def run_query(
        self,
        audio_batch: torch.tensor,
        text_batch: List[str],
        system_prompts: List[str],
        **kwargs,
    ) -> Tuple[torch.tensor, List, torch.tensor]:
        raise NotImplementedError("Subclasses must implement run_query method")

    def __call__(
        self,
        model_ids: str | tuple[str, ...],
        batch_prompt: BatchPrompt,
        print_prompt_and_response: bool,
        max_attempts_per_api_call: int,
        n: int,
        is_valid: Callable[[str], bool] = lambda _: True,
        **kwargs,
    ) -> list[LLMResponse]:
        assert len(model_ids) == 1, "Implementation only supports one model at a time."
        (model_id,) = model_ids

        start = time.time()
        audio_batch, text_batch, system_prompts = batch_prompt.batch_format()

        responses = []

        logprobs = None
        # Run query for uncached prompts
        api_start = time.time()
        output, decoded_output, logprobs = self.run_query(audio_batch, text_batch, system_prompts, **kwargs)
        api_duration = time.time() - api_start

        # Process and cache results for uncached prompts
        for i, completion in enumerate(decoded_output):
            response = LLMResponse(
                model_id=model_id,
                prompt=batch_prompt.prompts[i],
                stop_reason="max_tokens",
                completion=completion,
                cost=0.0,  # You might want to calculate the actual cost
                duration=time.time() - start,
                api_duration=api_duration / len(decoded_output),
                logprobs=logprobs,
            )
            responses.append(response)
        return responses


class DivaModel(BatchAudioModel):
    def __init__(self, model_name: str, prompt_history_dir: Path = None):
        super().__init__(model_name, prompt_history_dir)
        try:
            from alm_inference.submodules.DiVA.eval.models.via import ParallelVIA

            self.model = ParallelVIA("/workspace/pretrained_ckpts/DIVA/model-00001-of-00004.safetensors")
        except Exception as e:
            print(f"Error loading DiVA model: {e}")
            raise e

        self.kwarg_change_name = {
            "temperature": "temperature",
            "max_tokens": "max_new_tokens",
            "top_p": "top_p",
            "top_k": "top_k",
        }

    def run_query(
        self,
        audio_batch: torch.tensor,
        text_batch: List[str],
        system_prompts: List[str],
        **kwargs,
    ) -> Tuple[torch.tensor, List, torch.tensor]:
        params = {self.kwarg_change_name[k]: v for k, v in kwargs.items() if k in self.kwarg_change_name}
        return self.model.generate(audio_batch, text_batch, system_prompts, **params)
