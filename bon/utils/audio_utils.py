import base64
import io
import struct
import subprocess
import wave
from pathlib import Path
from string import Template
from subprocess import PIPE, Popen

import google.generativeai as genai
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
from pydub import AudioSegment
from vertexai.generative_models import Part


class WAVFile:
    """
    Custom class to read WAV files so we can get the header and audio data.
    The header is the bytes at the beginning of the file but it can be hard to extract
        without this complex logic.
    The audio data is the bytes after the header which are read as a numpy array in int16.
    """

    def __init__(self, audio: np.ndarray, header: bytes = None, metadata: dict = None):
        self.audio = audio
        self.header = header
        self.metadata = metadata
        self.sample_rate = 16000

        assert (self.metadata is None) == (
            self.header is None
        ), "Metadata and header must either both be None or both be provided"

        if self.metadata is None:
            self.metadata = {
                "riff": b"RIFF",
                "size": 44 + 2 * len(self.audio),  # 44 bytes for header, 2 bytes for each sample
                "format": b"WAVE",
                "aformat": 1,
                "channels": 1,
                "samplerate": self.sample_rate,
                "byterate": 32000,
                "blockalign": 2,
                "bps": 16,
            }

        if self.header is None:
            self.header = self.create_header(self.metadata)

    @classmethod
    def from_file(cls, file_path: str, override_sr: int = 16000):
        with open(file_path, "rb") as fh:
            header, audio, metadata = cls._read_wav(fh, override_sr)
        return cls(audio, header, metadata)

    @classmethod
    def from_bytes(cls, data: bytes, override_sr: int = 16000):
        with io.BytesIO(data) as fh:
            header, audio, metadata = cls._read_wav(fh, override_sr)
        return cls(audio, header, metadata)

    @staticmethod
    def _read_wav(fh, override_sr: int = 16000):
        metadata = {}
        with fh:
            riff, size, fformat = struct.unpack("<4sI4s", fh.read(12))
            metadata["riff"] = riff
            metadata["size"] = size
            metadata["format"] = fformat

            # Read header
            chunk_header = fh.read(8)
            subchunkid, subchunksize = struct.unpack("<4sI", chunk_header)

            if subchunkid == b"fmt ":
                aformat, channels, samplerate, byterate, blockalign, bps = struct.unpack("<HHIIHH", fh.read(16))
                metadata["aformat"] = aformat
                metadata["channels"] = channels
                metadata["samplerate"] = samplerate
                assert samplerate == override_sr, f"Samplerate does not match {override_sr}, got {samplerate}"
                metadata["byterate"] = byterate
                metadata["blockalign"] = blockalign
                metadata["bps"] = bps
                metadata["bitrate"] = (samplerate * channels * bps) / 1024

            chunkOffset = fh.tell()
            audio_data = None

            while chunkOffset < size:
                fh.seek(chunkOffset)
                subchunk2id, subchunk2size = struct.unpack("<4sI", fh.read(8))

                if subchunk2id == b"LIST":
                    listtype = struct.unpack("<4s", fh.read(4))
                    listOffset = 0
                    list_items = []
                    while (subchunk2size - 8) >= listOffset:
                        listitemid, listitemsize = struct.unpack("<4sI", fh.read(8))
                        listOffset = listOffset + listitemsize + 8
                        listdata = fh.read(listitemsize)
                        try:
                            listdata_decoded = listdata.decode("ascii")
                        except UnicodeDecodeError:
                            listdata_decoded = listdata.decode("utf-8", errors="replace")
                        list_items.append(
                            {
                                "listitemid": listitemid.decode("ascii"),
                                "listitemsize": listitemsize,
                                "listdata": listdata_decoded,
                            }
                        )
                    metadata["list"] = {"listtype": listtype, "list_items": list_items}
                elif subchunk2id == b"data":
                    header_data = fh.tell()
                    audio_data = np.frombuffer(fh.read(subchunk2size), dtype="<i2")
                    break  # Stop reading after finding the data chunk
                else:
                    fh.read(subchunk2size)

                chunkOffset = chunkOffset + subchunk2size + 8

            # Read the entire header as bytes
            fh.seek(0)
            header_bytes = fh.read(header_data)

            return header_bytes, audio_data, metadata

    @staticmethod
    def create_header(metadata: dict) -> bytes:
        # RIFF chunk
        header = struct.pack("<4sI4s", b"RIFF", metadata["size"], b"WAVE")  # 12 bytes

        # fmt sub-chunk
        subchunk1_size = 16
        header += struct.pack("<4sI", b"fmt ", subchunk1_size)  # 8 bytes
        header += struct.pack(
            "<HHIIHH",
            metadata["aformat"],
            metadata["channels"],
            metadata["samplerate"],
            metadata["byterate"],
            metadata["blockalign"],
            metadata["bps"],
        )  # 16 bytes

        # data sub-chunk
        data_size = metadata["size"] - 44
        header += struct.pack("<4sI", b"data", data_size)  # 8 bytes

        return header

    @property
    def duration(self) -> float:
        """
        Calculate and return the duration of the audio in seconds.
        """
        return len(self.audio) / self.metadata["samplerate"]

    def get_bytes(self):
        return self.header + self.audio.astype(np.int16).tobytes()

    def save_to_file(self, file_path: str):
        with open(file_path, "wb") as fh:
            fh.write(self.get_bytes())

    def print_metadata(self):
        for key, value in self.metadata.items():
            print(f"{key}: {value}")

    def write(self, path: str):
        wavfile.write(path, 16000, self.audio)


class IntermediateAugmentation:
    @staticmethod
    def slice_wav(wav_file: WAVFile, start: float, end: float) -> WAVFile:
        start_sample = int(start * wav_file.metadata["samplerate"])
        end_sample = int(end * wav_file.metadata["samplerate"])
        return WAVFile(wav_file.audio[start_sample:end_sample])

    @staticmethod
    def join_wav_files(*segments: WAVFile) -> WAVFile:
        joined_audio = np.concatenate([segment.audio for segment in segments])
        return WAVFile(joined_audio)

    @classmethod
    def apply(cls, wav_file: WAVFile, start: float, end: float, augment_func: callable) -> WAVFile:
        start_segment = cls.slice_wav(wav_file, 0, start)
        middle_segment = cls.slice_wav(wav_file, start, end)
        end_segment = cls.slice_wav(wav_file, end, len(wav_file.audio) / wav_file.metadata["samplerate"])

        augmented_middle = augment_func(middle_segment)

        return cls.join_wav_files(start_segment, augmented_middle, end_segment)


class SoxResample:
    RESAMPLE_CMD = Template("sox -t wav - -t wav -r $sr -")

    @classmethod
    def apply_sox_cmd_to_audio(cls, wav: WAVFile, override_sr: int = 16000) -> WAVFile:
        """
        Apply a sox command to the audio via subprocess. Sox is an audio processing tool that is used to apply various effects to audio files.
        We use WAVFile as input and output as it is convenient to handle both wavs on disk and byte streams that are returned by sox.
        """
        resample_cmd = cls.RESAMPLE_CMD.substitute(sr=override_sr)
        assert len(resample_cmd.strip()) > 0, f"Command is empty: {resample_cmd}"
        proc = Popen(resample_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = proc.communicate(wav.get_bytes())
        assert proc.returncode == 0, f"{resample_cmd} returned {stderr}"

        wav_out = WAVFile.from_bytes(stdout, override_sr=override_sr)
        return wav_out


def get_mime_type(audio_file) -> str:
    if audio_file[-4:] == ".wav":
        return "audio/wav"
    elif audio_file[-4:] == ".mp3":
        return "audio/mpeg"


def load_audio(audio_file: str, reader="librosa") -> bytes:
    if reader == "librosa":
        audio_data, sr = librosa.load(audio_file)
    elif reader == "soundfile":
        audio_data, sr = sf.read(audio_file)
    else:
        raise ValueError(f"Unknown reader: {reader}")
    with io.BytesIO() as buffer:
        sf.write(buffer, audio_data, sr, format="WAV")
        audio_bytes = buffer.getvalue()
    return audio_bytes


def get_audio_length(audio_file: str | Path) -> float:
    if isinstance(audio_file, Path):
        audio_path = audio_file
        audio_file = str(audio_file)
    else:
        audio_path = Path(audio_file)

    if audio_file == "":
        return 0
    if audio_path.suffix == ".wav":
        wav_out = WAVFile.from_file(audio_file)
    elif audio_path.suffix == ".mp3":
        wav_out = WAVFile.from_file(str(audio_path.with_suffix(".wav")))

    return wav_out.duration


def wav_to_float(x: np.ndarray) -> np.ndarray:
    """
    Input in range -2**15, 2**15 (or what is determined from dtype)
    Output in range -1, 1
    """
    max_value = np.iinfo(np.int16).max
    min_value = np.iinfo(np.int16).min
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
    x = x - min_value
    x = x / ((max_value - min_value) / 2.0)
    x = x - 1.0
    return x


def float_to_wav(x: np.ndarray) -> np.ndarray:
    """
    Input in range -1, 1
    Output in range -2**15, 2**15 (or what is determined from dtype)
    """
    max_value = np.iinfo(np.int16).max
    min_value = np.iinfo(np.int16).min

    x = x + 1.0
    x = x * (max_value - min_value) / 2.0
    x = x + min_value
    x = x.astype(np.int16)
    return x


def convert_to_wav_ffmpeg(input_path: str, output_path: str, sample_rate=16000):
    """
    Convert an audio file to 16-bit PCM WAV format using FFmpeg.

    :param input_path: Path to the input audio file
    :param output_path: Path to save the output WAV file
    :param sample_rate: Desired sample rate (default: 16000)
    :return: None
    """

    # assert output ends in .wav
    assert str(output_path).endswith(".wav"), f"Output path must end with .wav, got {output_path}"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        input_path,
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        output_path,
    ]
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise


def prepare_audio_part(audio_file: str, use_vertexai: bool = False) -> Part | genai.types.file_types.File:
    if use_vertexai:
        audio_bytes = load_audio(audio_file)
        return Part.from_data(data=audio_bytes, mime_type=get_mime_type(audio_file))
    else:
        audio = genai.upload_file(audio_file)
        return audio


def get_audio_data(audio_input: [str | Path | np.ndarray], return_WAVFile: bool = False) -> [WAVFile | np.ndarray]:
    sr = 16000
    if isinstance(audio_input, str):
        audio_input = Path(audio_input)
    if isinstance(audio_input, Path):
        if audio_input.suffix == ".wav":
            audio_file = audio_input
            # Check sample rate
            wav_audio = wave.open(str(audio_file), "rb")
            if wav_audio.getframerate() != 16000:
                sr = wav_audio.getframerate()
                print(f"Sampling rate of wavfile {audio_file} is {sr}")

        elif audio_input.suffix == ".mp3":
            wav_file = audio_input.with_suffix(".wav")
            if wav_file.exists():
                audio_file = wav_file
            else:
                # Convert file and save to disk and then load WAVFile from .wav file on disk
                convert_to_wav_ffmpeg(str(audio_input), str(wav_file))
                audio_file = wav_file
        try:
            if return_WAVFile:
                return WAVFile.from_file(audio_file, override_sr=sr)
            else:
                return WAVFile.from_file(audio_file, override_sr=sr).audio.astype(np.int16)
        except FileNotFoundError as f:
            print(f)
            return None

    elif isinstance(audio_input, np.ndarray):
        return audio_input.astype(np.int16)


def prepare_openai_s2s_audio(audio_file: [str | Path], target_sample_rate: int = 24000) -> str:
    """
    Format WAV or MP3 audio files for the OpenAI Realtime API.

    Args:
    audio_file_path (str): Path to the WAV or MP3 file.
    target_sample_rate (int): The desired sample rate (default is 24000 Hz).

    Returns:
    str: Base64 encoded PCM audio data.
    """
    wav_file = get_audio_data(audio_file, return_WAVFile=True)

    resampler = SoxResample()
    audio_24khz = resampler.apply_sox_cmd_to_audio(wav_file, override_sr=24000)

    # Ensure mono and 16-bit
    audio_24khz.audio = audio_24khz.audio[:, 0] if audio_24khz.audio.ndim > 1 else audio_24khz.audio
    audio_24khz.audio = audio_24khz.audio.astype(np.int16)

    # Convert audio to bytes
    audio_bytes = audio_24khz.get_bytes()
    # Encode bytes as base64 (this encodes to base64 and then the decode converts to a str which is what the API expects)
    base64_audio = base64.b64encode(audio_bytes).decode()

    return base64_audio


def create_silent_audio(audio_dir: Path, duration_ms: int = 3000) -> str:
    silent_file = audio_dir / f"silent_{duration_ms}ms.wav"
    if not silent_file.exists():
        silent_segment = AudioSegment.silent(duration=duration_ms)
        silent_segment.export(silent_file, format="wav")
    return str(silent_file)
