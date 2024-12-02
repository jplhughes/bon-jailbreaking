import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import httpx
import pandas as pd
from elevenlabs import save
from elevenlabs.client import AsyncElevenLabs
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm

from bon.data_models.hashable import deterministic_hash
from bon.utils.audio_utils import convert_to_wav_ffmpeg
from bon.utils.utils import setup_environment

from .voices import VOICE_ALIASES, VOICE_DICT

LOGGER = logging.getLogger(__name__)


async def generate_tts_audio_from_text(
    text: str,
    output_dir: Union[str, Path],
    voice: str = "Rachel",
    model: str = "eleven_multilingual_v2",
    client: AsyncElevenLabs = None,
    convert_to_wav: bool = False,
    name_override: Optional[str] = None,
) -> str:
    """
    Generate TTS audio from text and save it as a WAV file.

    :param text: Text to convert to speech
    :param output_dir: Path to save the output WAV file
    :param voice: Voice to use for TTS (default: "Rachel")
    :param model: Model to use for TTS (default: "eleven_multilingual_v2")
    :param client: ElevenLabs client (default: None)
    :param convert_to_wav: Whether to convert the audio to WAV format (default: False)
    :return: Path to the generated WAV file
    """
    assert voice in VOICE_DICT or voice in VOICE_ALIASES, f"{voice} is not in VOICE_DICT or VOICE_ALIASES!"
    if client is None:
        setup_environment()
        client = AsyncElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    hash_of_text = deterministic_hash(text)
    file_str = hash_of_text if name_override is None else name_override
    file_name = f"{file_str}.wav" if convert_to_wav else f"{file_str}.mp3"
    file_path = output_dir / file_name

    if file_path.exists():
        LOGGER.info(f"{file_path} already exists. Skipping generation")
        return str(file_path)

    if voice in VOICE_ALIASES:
        voice = VOICE_ALIASES[voice]

    def print_on_retry(retry_state):
        LOGGER.info(f"Retrying... Attempt {retry_state.attempt_number}")

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_random_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception_type(httpx.ConnectError),
        after=print_on_retry,
    )
    async def generate_audio_with_retry():
        audio = await client.generate(
            text=text,
            voice=VOICE_DICT[VOICE_ALIASES[voice]]["id"] if voice in VOICE_ALIASES else VOICE_DICT[voice]["id"],
            model=model,
        )
        audio_bytes = b""
        async for value in audio:
            audio_bytes += value
        return audio_bytes

    try:
        audio_bytes = await generate_audio_with_retry()
    except Exception as e:
        LOGGER.error(f"Error generating audio for text: {str(e)}")
        raise

    save(audio_bytes, file_path.with_suffix(".mp3"))
    if convert_to_wav:
        convert_to_wav_ffmpeg(file_path.with_suffix(".mp3"), file_path)
    LOGGER.info(f"Generated audio for text: {text}")

    return str(file_path)


async def generate_tts_audio_from_dataframe(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    transcription_col: str = "behavior",
    voice: str = "Rachel",
    model: str = "eleven_multilingual_v2",
    convert_to_wav: bool = False,
) -> List[str]:
    """
    Generate TTS audio for each row in a DataFrame and save as WAV files.

    :param df: DataFrame containing text to convert to speech
    :param output_dir: Directory to save the output WAV files
    :param transcription_col: Column name containing the text (default: "behavior")
    :param voice: Voice to use for TTS (default: "Rachel")
    :param model: Model to use for TTS (default: "eleven_multilingual_v2")
    :param convert_to_wav: Whether to convert the audio to WAV format (default: False)
    :return: List of paths to the generated WAV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "audio_file" not in df.columns:
        df["audio_file"] = ""

    setup_environment()
    client = AsyncElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    async def process_row(index: int, row: pd.Series, sem: asyncio.Semaphore) -> str:
        async with sem:
            text = row[transcription_col]
            try:
                file_path = await generate_tts_audio_from_text(
                    text, output_dir, voice, model, client=client, convert_to_wav=convert_to_wav
                )
                LOGGER.info(f"Generated audio for row {index}")
            except Exception as e:
                file_path = ""
                LOGGER.error(f"Error generating audio for row {index}: {str(e)}")

        return str(file_path)

    # Process rows concurrently
    sem = asyncio.Semaphore(15)
    tasks = [process_row(index, row, sem) for index, row in df.iterrows()]
    files_paths = await tqdm.gather(*tasks)
    return files_paths
