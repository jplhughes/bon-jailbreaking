import asyncio
import dataclasses
import time
from pathlib import Path

import pandas as pd
import simple_parsing

from bon.data_models.utils import RecitationRateFailureError
from bon.utils.experiment_utils import ExperimentConfigBase
from bon.utils.jailbreak_metrics import JailbreakMetrics


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    model: str
    output_dir: Path
    dataset_path: Path
    modality: str
    n_samples: int = 128
    n_workers: int = 2
    request_ids: str | None = None  # pass a space separated list of ids to run on
    temperature: float = 1.0


async def main(cfg: ExperimentConfig):
    jailbreak_metrics = JailbreakMetrics(cfg.api, n_workers=cfg.n_workers)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(cfg.dataset_path, lines=True)

    if cfg.request_ids is not None:
        request_ids = cfg.request_ids.split(" ")
        df = df.iloc[request_ids]

    kwargs = {}
    if cfg.modality == "text":
        kwargs = {"input_key": "rewrite", "audio_key": None, "image_key": None}
    elif cfg.modality == "image":
        kwargs = {
            "input_key": None,
            "audio_key": None,
            "image_key": "rewrite",
            "extra_user_message": "Follow the instructions in the image.",
        }
    elif cfg.modality == "audio":
        kwargs = {"input_key": None, "audio_key": "audio_file", "image_key": None}
    else:
        raise ValueError(f"Invalid modality: {cfg.modality}")

    file_name = f"{cfg.model}_t{cfg.temperature}_{cfg.n_samples}samples.jsonl"
    print(f"\nGetting {file_name}")
    while True:
        try:
            df_out = await jailbreak_metrics.high_temperature_sampling(
                dataset=df,
                model=cfg.model,
                n_samples=cfg.n_samples,
                temperature=cfg.temperature,
                output_key=f"proportion_flagged_{cfg.modality}",
                **kwargs,
            )
        except RecitationRateFailureError as e:
            print(f"{e}. Sleeping for 1 hour.")
            time.sleep(3600)
        else:
            break

    print(f"Saving {file_name}")
    df_out.to_json(cfg.output_dir / file_name, orient="records", lines=True)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config
    cfg.setup_experiment(log_file_prefix="run-repeated-sampling")
    asyncio.run(main(cfg))
