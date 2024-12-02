import asyncio
import dataclasses
import logging
import pathlib

import simple_parsing
from tqdm.auto import tqdm

from bon.utils.audio_utils import convert_to_wav_ffmpeg
from bon.utils.experiment_utils import ExperimentConfigBase

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    input_dir: pathlib.Path


async def main(cfg: ExperimentConfig):
    sem = asyncio.Semaphore(10)

    async def convert_file(file):
        LOGGER.info(f"converting {file}")
        async with sem:
            convert_to_wav_ffmpeg(file, file.with_suffix(".wav"))

    mp3_files = list(cfg.input_dir.glob("*.mp3"))
    wav_files = list(cfg.input_dir.glob("*.wav"))

    filter_files = [file for file in mp3_files if file.with_suffix(".wav") not in (wav_files)]
    LOGGER.info(f"Converting {len(filter_files)} new mp3 files to wavs")
    await tqdm.gather(*[convert_file(file) for file in filter_files])


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    cfg.setup_experiment(log_file_prefix="run-wav-conversion")
    asyncio.run(main(cfg))
