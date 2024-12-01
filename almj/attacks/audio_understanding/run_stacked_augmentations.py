import dataclasses
import itertools
import logging
import subprocess
from pathlib import Path
from typing import List

import simple_parsing

from almj.utils.experiment_utils import ExperimentConfigBase

LOGGER = logging.getLogger(__name__)


def parse_list(s: str, type_cast=float):
    if s is None:
        return None
    else:
        return [type_cast(item) for item in s.split(",")]


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    input_file_path: Path
    output_dir: Path
    RIRs_dir: str = "/mnt/jailbreak-defense/exp/data/RIRs"
    background_sound_dir: str = "./data"

    voice_filter: List[str] | None = None

    volumes: str | None = None
    pitch_shifts: str | None = None
    speeds: str | None = None
    music_snrs: str | None = None
    noise_snrs: str | None = None
    speech_snrs: str | None = None
    codecs: str | None = None
    reverberation_room_types: str | None = None

    n_augmentations: int = 2

    def parse_list(self, s: str, type_cast=float):
        if s is None:
            return None
        else:
            return [type_cast(item) for item in s.split(",")]

    def __post_init__(self):
        super().__post_init__()
        # update output_dir
        self.output_dir = self.output_dir / f"{self.n_augmentations}_augs"
        self.volumes = self.parse_list(self.volumes, float)
        self.pitch_shifts = self.parse_list(self.pitch_shifts, int)
        self.speeds = self.parse_list(self.speeds, float)
        self.music_snrs = self.parse_list(self.music_snrs, float)
        self.noise_snrs = self.parse_list(self.noise_snrs, float)
        self.speech_snrs = self.parse_list(self.speech_snrs, float)
        self.codecs = self.parse_list(self.codecs, str)
        self.reverberation_room_types = self.parse_list(self.reverberation_room_types, str)
        self.voice_filter = " ".join(self.voice_filter)

        assert any(
            v is not None
            for v in [
                self.volumes,
                self.pitch_shifts,
                self.speeds,
                self.music_snrs,
                self.noise_snrs,
                self.speech_snrs,
                self.codecs,
                self.reverberation_room_types,
            ]
        ), "At least one augmentation must be set"


def main(cfg: ExperimentConfig):

    # Define the augmentations as a dictionary
    augmentations = {
        key: value
        for key, value in {
            "volume": cfg.volumes,
            "pitch_shift": cfg.pitch_shifts,
            "speed": cfg.speeds,
            "music_snr": cfg.music_snrs,
            "noise_snr": cfg.noise_snrs,
            "speech_snr": cfg.speech_snrs,
            "codec": cfg.codecs,
            "reverberation_room_type": cfg.reverberation_room_types,
        }.items()
        if value is not None
    }

    # Get all the keys for different augmentations
    keys = list(augmentations.keys())

    # Generate all unique pairs of augmentation types
    all_combinations = list(itertools.combinations(keys, cfg.n_augmentations))

    for combination in all_combinations:
        # Get all possible values for each augmentation in the combination
        value_lists = [augmentations[aug] for aug in combination]

        # Iterate over all permutations of the current combination of augmentations
        for values in itertools.product(*value_lists):

            # Build the command for running augmentations
            command = [
                "python3",
                "-m",
                "almj.run.run_augmentations",
                "--input_file_path",
                cfg.input_file_path,
                "--output_dir",
                cfg.output_dir,
                "--voice_filter",
                cfg.voice_filter,
            ]

            # Add augmentation arguments to the command
            aug_list = []
            for aug, value in zip(combination, values):
                command.extend([f"--{aug}", str(value)])
                aug_list.append(f"{aug}__{value}")

            # Execute the command
            sub_dir = "__".join(aug_list)
            if (cfg.output_dir / sub_dir).exists():
                LOGGER.info(f"Skipping for augmentation {sub_dir} because they already exist")
            else:
                LOGGER.info(f"Running for: {' and '.join([f'{aug} {val}' for aug, val in zip(combination, values)])}")
                subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config
    # cfg.setup_experiment(log_file_prefix=cfg.augmentation_tag)
    main(cfg)
