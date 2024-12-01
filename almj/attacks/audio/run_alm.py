import asyncio
import dataclasses
import logging
import pathlib
import time
from typing import List

import pandas as pd
import simple_parsing
from tqdm.auto import tqdm

from almj.apis import InferenceAPI
from almj.data_models import Prompt
from almj.data_models.utils import RecitationRateFailureError
from almj.utils import utils
from almj.utils.experiment_utils import ExperimentConfigBase

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    """
    Generates ALM responses for audio files.
    """

    output_dir: pathlib.Path
    input_file: pathlib.Path
    model_ids: tuple[str] = ("gemini-1.5-flash-001", "gemini-1.5-pro-001")
    output_file: pathlib.Path = None
    gemini_use_vertexai: bool = False

    audio_field: str = "audio_file"
    text_field: str = "text_input"
    experiment_filter: List[str] | None = None

    temperature: float = 0.0
    max_tokens: int = 400
    n_samples: int = 1

    recitation_failure_sleep: int = 3600

    audio_out_dir: str | pathlib.Path = None  # We need to pass a file where speech out can be written

    def __post_init__(self):
        super().__post_init__()
        if self.output_file is None:
            self.output_file = self.output_dir / f"{self.input_file.stem}_responses.jsonl"
        if isinstance(self.model_ids, str):
            self.model_ids = (self.model_ids,)
        if "gpt-4o-s2s" in self.model_ids or self.model_ids == "gpt-4o-s2s":
            self.audio_out_dir = self.output_dir / "audio_out"
            self.audio_out_dir.mkdir(parents=True, exist_ok=True)
            assert self.audio_out_dir is not None, "gpt4o_audio_out_file is required when using gpt-4o-s2s model!"


def load_inputs(cfg: ExperimentConfig) -> list[dict]:
    print(cfg.input_file)
    df = pd.read_json(cfg.input_file, lines=True)

    if "gpt-4o-s2s" in cfg.model_ids or cfg.model_ids == "gpt-4o-s2s":
        df = df[df.experiment == "empty_transcript"]

    assert (
        len(df) > 0
    ), "No inputs with experiment='empty_transcript'. There must be at least one instance with experiment = 'empty_transcript' to run GPT-4o S2S model"

    if cfg.experiment_filter:
        LOGGER.info(f"Filtering for experiment: {cfg.experiment_filter}")
        assert all(
            experiment in df["experiment"].unique() for experiment in cfg.experiment_filter
        ), f"Experiment filter {cfg.experiment_filter} not in dataframe"
        df = df[df["experiment"].isin(cfg.experiment_filter)]

    return df.to_dict(orient="records")


async def get_standardised_model_response(
    input_obj: dict,
    model_id: str,
    api: InferenceAPI,
    gemini_use_vertexai: bool = False,
    audio_out_dir: str | pathlib.Path = None,
    temperature: float = 0.0,
    max_tokens: int = 400,
    n_samples: int = 1,
):
    prompt = Prompt.from_alm_input(
        audio_file=input_obj["audio_file"],
        user_prompt=input_obj["user_prompt"],
        system_prompt=input_obj["system_prompt"],
    )

    response = await api.__call__(
        model_ids=model_id,
        gemini_use_vertexai=gemini_use_vertexai,
        prompt=prompt,
        audio_out_dir=audio_out_dir,
        n=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    input_obj["model"] = model_id
    input_obj["temperature"] = temperature
    input_obj["max_tokens"] = max_tokens

    return [{**input_obj, **resp.to_dict()} for resp in response]


async def get_harmbench_model_response(
    input_obj: dict,
    audio_field: str,
    text_field: str,
    model_id: str,
    api: InferenceAPI,
    gemini_use_vertexai: bool = False,
    audio_out_dir: str | pathlib.Path = None,
    temperature: float = 0.0,
    max_tokens: int = 400,
    n_samples: int = 1,
):
    if text_field == "ignore":
        assert audio_field in input_obj, f"Audio field {audio_field} not found in input_obj"
        prompt = Prompt.from_alm_input(audio_file=input_obj[audio_field], user_prompt=None)
    elif audio_field == "ignore":
        assert text_field in input_obj, f"Text field {text_field} not found in input_obj"
        prompt = Prompt.from_alm_input(audio_file=None, user_prompt=input_obj[text_field])
    else:
        prompt = Prompt.from_alm_input(audio_file=input_obj[audio_field], user_prompt=input_obj[text_field])

    response = await api(
        model_ids=model_id,
        prompt=prompt,
        audio_out_dir=audio_out_dir,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_samples,
        gemini_use_vertexai=gemini_use_vertexai,
    )

    input_obj["alm_model"] = model_id
    input_obj["alm_temperature"] = temperature
    input_obj["alm_max_tokens"] = max_tokens

    return [{**input_obj, **resp.to_dict()} for resp in response]


async def main(cfg: ExperimentConfig):
    while True:
        try:
            inputs = load_inputs(cfg)

            responses: list[dict] = await tqdm.gather(
                *[
                    get_harmbench_model_response(
                        input_obj=input,
                        audio_field=cfg.audio_field,
                        text_field=cfg.text_field,
                        model_id=model,
                        api=cfg.api,
                        gemini_use_vertexai=cfg.gemini_use_vertexai,
                        audio_out_dir=cfg.audio_out_dir,
                        n_samples=cfg.n_samples,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens,
                    )
                    for input in inputs
                    for model in cfg.model_ids
                ]
            )

            # Flatten the responses
            flattened_responses = [item for sublist in responses for item in sublist]

            LOGGER.info(f"Saving responses to {cfg.output_file}")
            utils.save_jsonl(file_path=cfg.output_file, data=flattened_responses)

            return responses

        except RecitationRateFailureError as e:
            LOGGER.error(f"{str(e)}. Re-running the program in one hour.")
            time.sleep(cfg.recitation_failure_sleep)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config
    cfg.setup_experiment(log_file_prefix="run-audio-harmbench")

    asyncio.run(main(cfg))
