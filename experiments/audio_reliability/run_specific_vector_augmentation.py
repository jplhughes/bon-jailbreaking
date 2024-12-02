import asyncio
import dataclasses
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from simple_parsing import ArgumentParser

from bon.attacks.run_audio_bon import get_asr, get_augmentation_func
from bon.data_prep.augmentation import SoxAugmentation
from bon.utils.audio_utils import IntermediateAugmentation, WAVFile
from bon.utils.experiment_utils import ExperimentConfigBase
from bon.utils.shotgun_utils import process_single_shotgun


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    input_file_path: Path
    augmentations_path: Path | None = None
    chosen_augmentations: List[str] | None = None
    augmentation_vector: List[float] | None = None
    alm_model: str = "gemini-1.5-flash-001"
    alm_n_samples: int = 1
    alm_temperature: float = 1.0
    only_run_on_broken_example: bool = False
    use_same_file: bool = False
    behavior_id: str | None = None  # only run on a single behavior_id
    process_working_augs: bool = False

    # augmentation
    RIRs_dir: str | None = None
    background_sound_dir: str | None = None
    chosen_noise: str | None = None
    chosen_music: str | None = None
    chosen_speech: str | None = None

    prefix_file_path: str | None = None
    suffix_file_path: str | None = None

    extra_pitch_shift: float | None = None
    extra_speed_shift: float | None = None

    number_to_zero_out: int | None = None


async def apply_augmentation(
    wav_in_path: str,
    out_dir: str,
    augmentation_func: Callable,
    augmentation: SoxAugmentation,
    aug_sem: asyncio.Semaphore,
    prefix_file_path: str | None = None,
    suffix_file_path: str | None = None,
    extra_pitch_shift: float | None = None,
    extra_speed_shift: float | None = None,
):
    wav_out_path = Path(out_dir) / f"{Path(wav_in_path).stem}.wav"
    if wav_out_path.exists():
        return str(wav_out_path)
    async with aug_sem:
        wav_file = WAVFile.from_file(Path(wav_in_path).with_suffix(".wav"))

        # add prefix or suffix audio such as please and thanks
        files = []
        if prefix_file_path is not None:
            prefix_file_path = Path(prefix_file_path).with_suffix(".wav")
            prefix_file = WAVFile.from_file(prefix_file_path)
            files.append(prefix_file)
        files.append(wav_file)
        if suffix_file_path is not None:
            suffix_file_path = Path(suffix_file_path).with_suffix(".wav")
            suffix_file = WAVFile.from_file(suffix_file_path)
            files.append(suffix_file)
        if len(files) > 1:
            wav_file = IntermediateAugmentation.join_wav_files(*files)

        # extra augmentation to test brittleness
        if extra_pitch_shift is not None:
            print(f"Applying extra pitch shift of {extra_pitch_shift}")
            wav_file = augmentation.apply_shift_pitch(wav_file, extra_pitch_shift)

        if extra_speed_shift is not None:
            print(f"Applying extra speed shift of {extra_speed_shift}")
            wav_file = augmentation.apply_speed(wav_file, extra_speed_shift)

        wav_file = augmentation_func(wav_file)
        wav_file.write(wav_out_path)
        return str(wav_out_path)


async def process_audio_files(
    cfg: ExperimentConfig,
    chosen_augmentations: List[str],
    augmentation_vector: List[float],
    df: pd.DataFrame,
    audio_file_dir: Path,
    augmentation: SoxAugmentation,
):

    # maybe zero out the lowest N values in the vector
    if cfg.number_to_zero_out is not None:
        assert cfg.number_to_zero_out < len(
            chosen_augmentations
        ), "number_to_zero_out needs to be less than the number of augmentations"
        # sort the vector and get the index of smallest N
        sorted_indices = np.argsort(np.abs(augmentation_vector))
        smallest_two_indices = sorted_indices[: cfg.number_to_zero_out]
        for i in smallest_two_indices:
            print(f"Zeroing out {augmentation_vector[i]} for augmentation {chosen_augmentations[i]}")
            augmentation_vector[i] = 0.0

    augmentation_func, vector_augmentation = get_augmentation_func(
        chosen_augmentations, augmentation_vector, augmentation, cfg.chosen_music, cfg.chosen_noise, cfg.chosen_speech
    )

    specific_audio_dir = audio_file_dir / vector_augmentation.file_name()
    specific_audio_dir.mkdir(parents=True, exist_ok=True)

    aug_sem = asyncio.Semaphore(5)

    if cfg.use_same_file:
        assert len(df) == 1, "If use_same_file is True, df must have only one row"
        assert cfg.only_run_on_broken_example, "If use_same_file is True, only_run_on_broken_example must be True"
        augmented_files = [df.iloc[0]["augmented_file"]]
        # save file path to txt file
        with open(specific_audio_dir / "file_paths.txt", "w") as f:
            f.write(augmented_files[0])
    else:
        augmented_files = await asyncio.gather(
            *[
                apply_augmentation(
                    row["audio_file"],
                    specific_audio_dir,
                    augmentation_func,
                    augmentation,
                    aug_sem,
                    cfg.prefix_file_path,
                    cfg.suffix_file_path,
                    cfg.extra_pitch_shift,
                    cfg.extra_speed_shift,
                )
                for _, row in df.iterrows()
            ]
        )
    asr = await get_asr(
        specific_audio_dir,
        cfg.alm_model,
        cfg.api,
        df,
        augmented_files,
        n_samples=cfg.alm_n_samples,
        alm_temperature=cfg.alm_temperature,
    )
    return asr, vector_augmentation, specific_audio_dir


async def main(cfg: ExperimentConfig):
    df = pd.read_json(cfg.input_file_path, lines=True)
    if cfg.behavior_id is not None:
        df = df[df["behavior_id"] == cfg.behavior_id]
        assert len(df) == 1, f"Behavior ID not found in df: {cfg.behavior_id}"
    audio_file_dir = Path(cfg.output_dir) / "audio_files"
    audio_file_dir.mkdir(parents=True, exist_ok=True)
    augmentation = SoxAugmentation(RIRs_dir=cfg.RIRs_dir, background_sound_dir=cfg.background_sound_dir)

    if cfg.chosen_augmentations is not None and cfg.augmentation_vector is not None:
        asr, vector_augmentation, _ = await process_audio_files(
            cfg, cfg.chosen_augmentations, cfg.augmentation_vector, df, audio_file_dir, augmentation
        )
        print(f"ASR: {asr}, Augmentation Details: {vector_augmentation}")
    elif cfg.augmentations_path is not None:
        df_augs = pd.read_json(cfg.augmentations_path, lines=True)
        if cfg.process_working_augs:
            df_augs = process_single_shotgun(cfg.augmentations_path, "exp", cfg.alm_model, df, 159)
            df_augs = df_augs[df_augs.asr > 0]

        for index, row in df_augs.iterrows():
            print(f"Processing vector that broke: {row['direct_request']}\nVector={row['vector']}")
            chosen_augmentations = row["augs"].split(",")
            augmentation_vector = row["vector"]
            if cfg.only_run_on_broken_example:
                assert "direct_request" in row, "direct_request not found in row"
                df1 = df[df["rewrite"] == row["direct_request"]]
                if cfg.use_same_file:
                    assert "augmented_file" in row, "augmented_file not found in row"
                    df1["augmented_file"] = row["augmented_file"]
                assert len(df1) == 1, "Example not found in df for direct_request"
            else:
                df1 = df
            asr, vector_augmentation, specific_audio_dir = await process_audio_files(
                cfg, chosen_augmentations, augmentation_vector, df1, audio_file_dir, augmentation
            )
            df_augs.at[index, "alm_response_path"] = str(specific_audio_dir / "alm_responses.json")
            df_augs.at[index, "asr"] = asr
            # Save intermediate checkpoint after each iteration
            df_augs.to_json(cfg.output_dir / "results.jsonl", lines=True, orient="records")
            print(f"ASR: {asr*100:.2f} for vector that broke {row['direct_request']}")
            print(f"Saved intermediate checkpoint for vector {index + 1}/{len(df_augs)}")
    else:
        print("Error: Either chosen_augmentations and augmentation_vector or augmentations_path must be provided.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config
    cfg.setup_experiment(log_file_prefix="run_specific_vector_augmentation")
    asyncio.run(main(cfg))
