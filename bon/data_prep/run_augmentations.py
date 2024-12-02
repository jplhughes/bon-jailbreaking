import dataclasses
import logging
from pathlib import Path
from typing import List

import pandas as pd
import scipy.io.wavfile as wavfile
import simple_parsing

from bon.data_prep.augmentation import SoxAugmentation
from bon.utils.audio_utils import WAVFile
from bon.utils.experiment_utils import ExperimentConfigBase

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    input_file_path: Path
    output_dir: Path
    RIRs_dir: str = "/mnt/jailbreak-defense/exp/data/RIRs"
    background_sound_dir: str = "./data"
    colored_noise_dir: str = "/mnt/jailbreak-defense/exp/data/colored_noise"

    voice_filter: List[str] | None = None
    attack_filter: List[str] | None = None

    volume: float | None = None
    pitch_shift: int | None = None
    speed: float | None = None
    reverberation_room_type: str | None = None
    telephony_codec: str | None = None

    music_snr: float | None = None
    music_path: str | None = None
    noise_snr: float | None = None
    noise_path: str | None = None
    speech_snr: float | None = None
    speech_path: str | None = None

    colored_noise_type: str | None = None
    colored_noise_snr: float | None = None

    def __post_init__(self):
        super().__post_init__()
        # at least one augmentation must be set
        assert any(
            v is not None
            for v in [
                self.volume,
                self.pitch_shift,
                self.reverberation_room_type,
                self.telephony_codec,
                self.music_snr,
                self.noise_snr,
                self.speech_snr,
                self.speed,
                self.colored_noise_snr,
            ]
        ), "At least one augmentation must be set"
        self.augmentation_tag = "__".join(
            [
                f"{k}__{v}"
                for k, v in self.__dict__.items()
                if k
                in [
                    "volume",
                    "pitch_shift",
                    "reverberation_room_type",
                    "telephony_codec",
                    "music_snr",
                    "noise_snr",
                    "speech_snr",
                    "speed",
                    "colored_noise_snr",
                ]
                and v is not None
            ]
        )

        self.augmentation_tag = self.augmentation_tag.replace(
            "colored_noise_snr", f"{self.colored_noise_type}_noise_snr"
        )

        if self.music_path is not None:
            self.augmentation_tag = str.split(self.music_path, "/")[-1][:-4] + "_snr__" + str(self.music_snr)

        elif self.noise_path is not None:
            self.augmentation_tag = str.split(self.noise_path, "/")[-1][:-4] + "_snr__" + str(self.noise_snr)

        elif self.speech_path is not None:
            self.augmentation_tag = str.split(self.speech_path, "/")[-1][:-4] + "_snr__" + str(self.speech_snr)


def main(cfg: ExperimentConfig):
    # output setup
    output_dir = Path(cfg.output_dir) / cfg.augmentation_tag
    audio_file_dir = output_dir / "audio_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_file_dir.mkdir(parents=True, exist_ok=True)

    orig_name = Path(cfg.input_file_path).stem
    output_input_file_path = output_dir / f"{orig_name}_augmented.jsonl"
    if output_input_file_path.exists():
        LOGGER.info(f"Skipping {output_input_file_path} because it already exists")
        return

    # input df and filter
    df = pd.read_json(cfg.input_file_path, lines=True)
    if cfg.voice_filter:
        LOGGER.info(f"Filtering for voice: {cfg.voice_filter}")
        assert all(
            voice in df["voice"].unique() for voice in cfg.voice_filter
        ), f"Voice filter {cfg.voice_filter} not in dataframe"
        df = df[df["voice"].isin(cfg.voice_filter)]
    if cfg.attack_filter:
        LOGGER.info(f"Filtering for attack: {cfg.attack_filter}")
        assert all(
            attack in df["attack"].unique() for attack in cfg.attack_filter
        ), f"Attack filter {cfg.attack_filter} not in dataframe"
        df = df[df["attack"].isin(cfg.attack_filter)]

    augmentation = SoxAugmentation(
        RIRs_dir=cfg.RIRs_dir, background_sound_dir=cfg.background_sound_dir, colored_noise_dir=cfg.colored_noise_dir
    )

    for idx, row in df.iterrows():
        audio_file = Path(row["audio_file"]).with_suffix(".wav")
        wav_file = WAVFile.from_file(audio_file)

        LOGGER.info(f"Applying augmentations: {cfg.augmentation_tag} to {audio_file}")

        wav_out_path = audio_file_dir / f"{audio_file.stem}.wav"
        if wav_out_path.exists():
            LOGGER.info(f"Skipping {wav_out_path} because it already exists")
            df.at[idx, "audio_file"] = str(wav_out_path)
            continue

        if cfg.volume is not None:
            wav_file = augmentation.apply_vol_pertubation(wav_file, vol=cfg.volume)
        if cfg.telephony_codec is not None:
            wav_file = augmentation.apply_8khz_telephony(wav_file, codec=cfg.telephony_codec)
        if cfg.reverberation_room_type is not None:
            wav_file = augmentation.apply_reverberation(wav_file, room_type=cfg.reverberation_room_type)
        if cfg.music_snr is not None:
            wav_file = augmentation.apply_music(wav_file, snr=cfg.music_snr, music_path=cfg.music_path)
        if cfg.noise_snr is not None:
            wav_file = augmentation.apply_noise(wav_file, snr=cfg.noise_snr, noise_path_override=cfg.noise_path)
        if cfg.speech_snr is not None:
            wav_file = augmentation.apply_speech(wav_file, snr=cfg.speech_snr, speech_path=cfg.speech_path)
        if cfg.speed is not None:
            wav_file = augmentation.apply_speed(wav_file, speed=cfg.speed)
        if cfg.colored_noise_snr is not None:
            wav_file = augmentation.apply_colored_noise(
                wav_file, noise_type=cfg.colored_noise_type, snr=cfg.colored_noise_snr
            )
        if cfg.pitch_shift is not None:
            wav_file = augmentation.apply_shift_pitch(wav_file, pitch_shift=cfg.pitch_shift)

        wavfile.write(wav_out_path, 16000, wav_file.audio)
        df.at[idx, "audio_file"] = str(wav_out_path)

    df["augmentation_tag"] = cfg.augmentation_tag
    df["volume"] = cfg.volume
    df["pitch_shift"] = cfg.pitch_shift
    df["reverberation_room_type"] = cfg.reverberation_room_type
    df["telephony_codec"] = cfg.telephony_codec
    df["music_snr"] = cfg.music_snr
    df["noise_snr"] = cfg.noise_snr
    df["speech_snr"] = cfg.speech_snr
    df["music_path"] = cfg.music_path
    df["noise_path"] = cfg.noise_path
    df["speech_path"] = cfg.speech_path
    df["speed"] = cfg.speed
    df[f"{cfg.colored_noise_type}_noise_snr"] = cfg.colored_noise_snr

    # Save updated dataframe to new JSONL file
    df.to_json(output_input_file_path, orient="records", lines=True)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    cfg.setup_experiment(log_file_prefix=cfg.augmentation_tag)
    main(cfg)
