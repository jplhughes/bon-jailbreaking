import asyncio
import dataclasses
import logging
import re
from pathlib import Path
from typing import List

import pandas as pd
import simple_parsing

from bon.apis.tts.elevenlabs import generate_tts_audio_from_dataframe, generate_tts_audio_from_text
from bon.apis.tts.voices import VOICE_ALIASES, VOICE_DICT
from bon.utils.experiment_utils import ExperimentConfigBase

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class TTSExperimentConfig(ExperimentConfigBase):
    """
    Configuration for processing voices and generating TTS audio.
    """

    input_file: Path | None = None
    output_file: Path | None = None
    audio_file_dir: Path | None = None
    voices: str | List[str] | None = None
    model: str | None = "eleven_multilingual_v2"
    harmbench_style: bool = True
    silence: bool = False
    harmbench_path: Path | None = None
    jsonl_dir: Path | None = None
    transcription_col: str | None = "rewrite"
    add_experiment_variants: bool | None = True

    def __post_init__(self):
        super().__post_init__()
        if self.harmbench_style:
            assert self.harmbench_path is not None, "harmbench_path must be provided if harmbench_style is True"
            assert self.jsonl_dir is not None, "jsonl_dir must be provided if harmbench_style is True"
            assert self.audio_file_dir is not None, "audio_file_dir must be provided if harmbench_style is True"
            assert self.voices is not None, "voices must be provided if harmbench_style is True"
            assert self.model is not None, "model must be provided if harmbench_style is True"
        else:
            assert self.input_file is not None, "input_file must be provided if harmbench_style is False"
            assert self.output_file is not None, "output_file must be provided if harmbench_style is False"
            assert self.audio_file_dir is not None, "audio_file_dir must be provided if harmbench_style is False"
            assert self.voices is not None, "voices must be provided if harmbench_style is False"
            assert self.model is not None, "model must be provided if harmbench_style is False"

        if isinstance(self.voices, str):
            if " " in self.voices:
                self.voices = str.split(self.voices, " ")
            else:
                self.voices = [self.voices]


async def process_voices(
    jsonl_dir: Path,
    audio_file_dir: Path,
    harmbench_path: Path,
    voices: list[str],
    transcription_col: str = "rewrite",
    audio_request: str = "Please answer the request.",
):

    df = pd.read_json(harmbench_path, lines=True)
    dfs = []
    for voice in voices:
        file_paths = await generate_tts_audio_from_dataframe(
            df, audio_file_dir / f"elevenlabs/{voice}", voice=voice, transcription_col=transcription_col
        )
        df_ = df.copy()
        df_["audio_file"] = file_paths
        df_["voice"] = voice

        if voice in VOICE_ALIASES:
            voice_info = VOICE_DICT[VOICE_ALIASES[voice]]
        else:
            voice_info = VOICE_DICT[voice]
        df_["voice_gender"] = voice_info["gender"]
        df_["voice_age"] = voice_info["age"]
        df_["voice_nationality"] = voice_info["nationality"]
        df_["voice_type"] = voice_info["voice"]
        dfs.append(df_)

        name_override = re.sub(r"[^\w\s-]", "", audio_request).replace(" ", "-").lower()
        await generate_tts_audio_from_text(
            audio_request,
            audio_file_dir / f"elevenlabs/{voice}",
            voice=voice,
            name_override=name_override,
        )

    # silent 2s audio file
    silence_path = audio_file_dir / "silence.wav"
    if not silence_path.exists():
        create_silent_file(silence_path)

    df = pd.concat(dfs)
    df.to_json(jsonl_dir / "harmbench_audio.jsonl", orient="records", lines=True)

    df_all = add_experiment_variants(
        df, audio_file_dir, silence_path, transcription_col=transcription_col, audio_request_name=name_override
    )

    df_all.to_json(jsonl_dir / "harmbench_audio_experiments.jsonl", orient="records", lines=True)


def add_experiment_variants(
    df: pd.DataFrame,
    audio_file_dir: Path,
    silence_path: Path,
    transcription_col: str,
    audio_request_name: str,
    experiment_types: list[str] = [
        "transcript_provided",
        "empty_transcript",
        "text_prompt_to_answer",
        "audio_prompt_to_answer",
        "text_only",
    ],
) -> pd.DataFrame:
    possible_experiment_types = [
        "transcript_provided",
        "empty_transcript",
        "text_prompt_to_answer",
        "audio_prompt_to_answer",
        "silence_audio_2s",
        "text_only",
    ]
    assert all(
        exp in possible_experiment_types for exp in experiment_types
    ), f"Invalid experiment type(s) {[exp for exp in experiment_types if exp not in possible_experiment_types]}"
    df1 = df.copy()
    df1["text_input"] = df1[transcription_col]
    df1["experiment"] = "transcript_provided"

    df2 = df.copy()
    df2["text_input"] = ""
    df2["experiment"] = "empty_transcript"

    df3 = df.copy()
    df3["text_input"] = "Please answer the request."
    df3["experiment"] = "text_prompt_to_answer"

    df4 = df.copy()
    df4["text_input"] = df4[transcription_col]
    df4["audio_file"] = df4.apply(
        lambda row: str(audio_file_dir / f"elevenlabs/{row['voice']}/{audio_request_name}.mp3"), axis=1
    )
    df4["experiment"] = "audio_prompt_to_answer"

    df5 = df.copy()
    df5["text_input"] = df5[transcription_col]
    df5["audio_file"] = str(silence_path)
    df5["experiment"] = "silence_audio_2s"

    df6 = df.copy()
    df6["text_input"] = df6[transcription_col]
    df6["audio_file"] = ""
    df6["experiment"] = "text_only"

    df_all = pd.concat([df1, df2, df3, df4, df5, df6])
    df_all = df_all[df_all["experiment"].isin(experiment_types)]
    print(f"Returning experiment df with {len(df_all)} rows and variants for {experiment_types}")
    return df_all


def create_silent_file(silence_path: Path):
    import wave

    import numpy as np

    silence_path.parent.mkdir(parents=True, exist_ok=True)

    duration = 2  # seconds
    sample_rate = 16000  # Hz
    audio = np.zeros(int(duration * sample_rate), dtype=np.int16)
    with wave.open(str(silence_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


async def process_voice(
    input_file: Path,
    audio_file_dir: Path,
    output_file: Path,
    voice: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel (Legacy)
    model: str = "eleven_multilingual_v2",
    transcription_col: str = "transcript",
    audio_request: str = "Please answer the request.",
):

    df = pd.read_json(input_file, lines=True)

    audio_file_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(voice, list):
        voice = voice[0]

    audio_file_paths = await generate_tts_audio_from_dataframe(
        df=df,
        output_dir=audio_file_dir / f"elevenlabs/{voice}",
        transcription_col=transcription_col,
        voice=voice,
        model=model,
        convert_to_wav=True,
    )
    df["audio_file"] = audio_file_paths
    df["voice"] = voice
    df.to_json(output_file, orient="records", lines=True)

    name_override = re.sub(r"[^\w\s-]", "", audio_request).replace(" ", "-").lower()
    # silent 2s audio file
    silence_path = audio_file_dir / "silence.wav"
    if not silence_path.exists():
        create_silent_file(silence_path)

    df.to_json(output_file, orient="records", lines=True)

    if cfg.add_experiment_variants:
        df_all = add_experiment_variants(
            df, audio_file_dir, silence_path, transcription_col=transcription_col, audio_request_name=name_override
        )

        file_name = output_file.stem
        all_exp_path = str.replace(str(output_file), file_name, file_name + "_experiments")
        df_all.to_json(all_exp_path, orient="records", lines=True)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(TTSExperimentConfig, dest="tts_config")
    args = parser.parse_args()

    cfg: TTSExperimentConfig = args.tts_config
    cfg.setup_experiment(log_file_prefix="run-tts")

    if cfg.harmbench_style:
        assert cfg.harmbench_path is not None, "harmbench_path must be provided if harmbench_style is True"
        if isinstance(cfg.voices, str):
            cfg.voices = [cfg.voices]
        asyncio.run(
            process_voices(
                jsonl_dir=cfg.output_file.parent,
                audio_file_dir=cfg.audio_file_dir,
                harmbench_path=cfg.harmbench_path,
                voices=cfg.voices,
                transcription_col=cfg.transcription_col,
            )
        )
    else:
        if cfg.silence:
            silence_path = cfg.audio_file_dir / "silent_file.wav"
            create_silent_file(silence_path)
            df = pd.read_json(cfg.input_file, lines=True)
            df["audio_file"] = str(silence_path)
            df["voice"] = "silent"
            df.to_json(cfg.output_file, orient="records", lines=True)
        else:
            asyncio.run(
                process_voice(
                    input_file=cfg.input_file,
                    output_file=cfg.output_file,
                    voice=cfg.voices,
                    audio_file_dir=cfg.audio_file_dir,
                    model=cfg.model,
                    transcription_col=cfg.transcription_col,
                )
            )
