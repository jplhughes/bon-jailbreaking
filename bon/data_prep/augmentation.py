import logging
import random
from pathlib import Path
from string import Template
from subprocess import PIPE, Popen

import augment
import numpy as np
import torch

from bon.utils.audio_utils import WAVFile, float_to_wav, wav_to_float

TARGET_SAMPLING_RATE = 16000
DTYPE = "<i2"


def quick_choice(list: list):
    return list[random.randint(0, len(list) - 1)]


def process_rirs_dir(RIRs_dir: str):
    RIRs_data = {
        "small_room": [],
        "medium_room": [],
        "large_room": [],
        "real_isotropic": [],
    }
    RIRs_files = {
        "small_room": f"{RIRs_dir}/simulated_rirs/smallroom/rir_list",
        "medium_room": f"{RIRs_dir}/simulated_rirs/mediumroom/rir_list",
        "large_room": f"{RIRs_dir}/simulated_rirs/largeroom/rir_list",
        "real_isotropic": f"{RIRs_dir}/real_rirs_isotropic_noises/rir_list",
    }
    # extract paths of each RIR and ignore other metadata
    for room_type, RIRs_list in RIRs_files.items():
        with open(RIRs_list) as RIR_fp:
            for RIR_entry in RIR_fp:
                RIR = RIR_entry.split()[4]
                RIRs_data[room_type].append(f"{RIRs_dir}/{RIR}")

    return RIRs_data


def process_sound_dblx(dblx_path: str, background_sound_dir: str):
    sound_paths = []
    with open(dblx_path, "r") as in_f:
        for line in in_f:
            path, dur = line.split()
            sound_paths.append((f"{background_sound_dir}/{path}", float(dur)))
    return sound_paths


class SoxAugmentation:
    """This is an augmentation class that manages the behaviour of applying the following:
        WavAugment - pitch and reverb
        Kaldi style - RIRs and volume perturbation

    Args:
        RIRs_dir (string): path to RIRs dir for kaldi-style augmentation
        background_sound_dir (string): path to background sounds dir
        colored_noise_dir (string): path to colored noise dir
    """

    RIRs_CMD = Template(
        '/opt/kaldi/src/featbin/wav-reverberate --shift-output=true --impulse-response="sox $RIR -r $sr -t wav - |"  - -'
    )
    VOL_CMD = Template("sox --vol $vol -t wav - -t wav -")

    RESAMPLE_CMD = Template("sox -t wav - -t wav -r $sr -")
    TELEPHONY_CMD = Template("sox -v 0.9 -t wav - -t wav -R -e $codec - | sox -t wav - -t wav -R -e signed -b 16 -")
    BACKGROUND_NOISE_CMD = Template(
        "/opt/kaldi/src/featbin/wav-reverberate --shift-output=true --additive-signals='"
        "$sounds ' --start-times=$start_times --snrs=$snrs - -"
    )
    SPEED_CMD = Template("sox -t wav - -t wav - tempo $speed")
    TRIM_CMD = Template("sox -t wav - -t wav - trim $start =$end")
    COLOR_NOISE_CMD = Template("sox -m -v 1.0 - -v $noise_volume $noise_file -t wav - trim 0 $duration")

    def __init__(self, RIRs_dir=None, background_sound_dir=None, colored_noise_dir=None):
        # RIRs
        self.RIRs_dir = RIRs_dir
        if RIRs_dir is not None and Path(RIRs_dir).exists():
            logging.info(f"Using RIRs for reverb augmentation from {RIRs_dir}")
            self.RIRs_data = process_rirs_dir(RIRs_dir)
        else:
            logging.warning(f"RIRs directory {RIRs_dir} does not exist, not using RIRs for reverb augmentation")
            self.RIRs_data = None

        # Background sounds
        self.background_sound_dir = background_sound_dir
        if background_sound_dir is not None and Path(background_sound_dir).exists():
            logging.info(f"Using background sounds from {background_sound_dir}")
            self.background_music = process_sound_dblx("./data/music.dblx", background_sound_dir)
            self.background_speech = process_sound_dblx("./data/speech.dblx", background_sound_dir)
            self.background_noises = process_sound_dblx("./data/noise.dblx", background_sound_dir)
        else:
            logging.warning(
                f"Background sound directory {background_sound_dir} does not exist, not using background sounds"
            )
            self.background_music = None
            self.background_speech = None
            self.background_noises = None
            self.background_sound_dir = None

        if colored_noise_dir is not None and Path(colored_noise_dir).exists():
            self.colored_noise_files = {
                color: Path(colored_noise_dir) / f"{color}_noise.wav"
                for color in ["white", "pink", "brown", "blue", "violet", "gray"]
            }

    def apply_shift_pitch(self, wav: WAVFile, pitch_shift: int) -> WAVFile:
        """
        Shift the pitch of the audio by the given amount (-300 to 300 is recommended).
        """
        x = wav.audio
        x_orig = x.copy()

        x = wav_to_float(x)
        x = np.expand_dims(x, axis=0)

        src_info = {
            "channels": x.shape[0],
            "length": x.shape[1],
            "precision": 32,
            "rate": TARGET_SAMPLING_RATE,
            "bits_per_sample": 32,
        }

        target_info = {
            "channels": 1,
            "length": x.shape[1],
            "precision": 32,
            "rate": TARGET_SAMPLING_RATE,
            "bits_per_sample": 32,
        }

        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_tensor = (
            augment.EffectChain()
            .pitch("-q", pitch_shift)
            .rate(TARGET_SAMPLING_RATE)
            .apply(x_tensor, src_info=src_info, target_info=target_info)
        )

        x = x_tensor.numpy()

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if np.isnan(x).any() or np.isinf(x).any():
            logging.warn("Sox and pitch augmentation gave nan/inf, skipping augmnetation for file")
            return x_orig

        x = np.squeeze(x, axis=0)
        x = float_to_wav(x)

        x = WAVFile(x)

        return x

    @staticmethod
    def apply_sox_cmd_to_audio(wav: WAVFile, cmd: str) -> WAVFile:
        """
        Apply a sox command to the audio via subprocess. Sox is an audio processing tool that is used to apply various effects to audio files.
        We use WAVFile as input and output as it is convenient to handle both wavs on disk and byte streams that are returned by sox.
        """
        assert len(cmd.strip()) > 0, f"Command is empty: {cmd}"
        proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = proc.communicate(wav.get_bytes())
        assert proc.returncode == 0, f"{cmd} returned {stderr}"

        wav_out = WAVFile.from_bytes(stdout)
        return wav_out

    def apply_reverberation(
        self,
        wav: WAVFile,
        sampling_rate: int = 16000,
        room_type: str = "small_room",
        reverb_file: str = None,
        file_id: int = None,
    ) -> WAVFile:
        """
        Apply a reverberation effect to the audio. We use the RIRs from the Kaldi RIRs dataset.
        There are RIRs that simulate small, medium and large rooms.
        """
        assert self.RIRs_dir is not None, "RIRs directory is not set"
        assert room_type in self.RIRs_data, f"Invalid room type: {room_type}"
        if file_id is not None:
            assert reverb_file is None, "Cannot specify both file_id and reverb_file"
            file_id = file_id % len(self.RIRs_data[room_type])
            reverb_file = self.RIRs_data[room_type][file_id]
        if reverb_file is None:
            reverb_file = quick_choice(self.RIRs_data[room_type])
        assert reverb_file in self.RIRs_data[room_type], f"Invalid reverb_file: {reverb_file}"

        RIRs_cmd = self.RIRs_CMD.substitute(RIR=reverb_file, sr=sampling_rate)
        wav_out = self.apply_sox_cmd_to_audio(wav, cmd=RIRs_cmd)
        return wav_out

    def apply_vol_pertubation(self, wav: WAVFile, vol: float) -> WAVFile:
        """
        Apply a volume perturbation to the audio. 0.125 to 2.0 is the range of kaldi defaults
        """
        vol_cmd = self.VOL_CMD.substitute(vol=vol)
        wav_out = self.apply_sox_cmd_to_audio(wav, cmd=vol_cmd)
        return wav_out

    def apply_8khz_telephony(self, wav: WAVFile, sampling_rate: int = 16000, codec: str = "u-law") -> WAVFile:
        """
        Simulate 8kHz telephony by downsampling to 8kHz, applying phone codec ("u-law", "ima-adpcm") and upsampling to 16kHz
        """
        assert codec in ["u-law", "ima-adpcm", "a-law"], f"Invalid codec: {codec}"
        downsample_line = self.RESAMPLE_CMD.substitute(sr=8000)
        codec_line = self.TELEPHONY_CMD.substitute(codec=codec)
        upsample_line = self.RESAMPLE_CMD.substitute(sr=sampling_rate)
        telephony_8khz_cmd = f"{downsample_line} | {codec_line} | {upsample_line}"

        wav_out = self.apply_sox_cmd_to_audio(wav, cmd=telephony_8khz_cmd)
        return wav_out

    def apply_noise(
        self, wav: WAVFile, snr: int = None, noise_path_override: str = None, file_id: int = None
    ) -> WAVFile:
        """
        Apply background noise to the audio at 1s intervals. We use noises from the MUSAN dataset.
        If noise_path is None, we will randomly pick a noise from the MUSAN dataset which will be
        different at each interval.
        """
        dur = round(wav.audio.shape[0] / TARGET_SAMPLING_RATE, 2)

        sounds, start_times, snrs = [], [], []
        total_noise_dur = 0
        if noise_path_override is None:
            assert self.background_sound_dir is not None, "Background sound directory is not set"

        # keep adding noise until the total duration is greater than the audio duration
        while total_noise_dur < dur:
            if file_id is not None:
                assert noise_path_override is None, "Cannot specify both file_id and noise_path_override"
                file_id = file_id % len(self.background_noises)
                noise_path, noise_dur = self.background_noises[file_id]
            elif noise_path_override is None:
                noise_path, noise_dur = quick_choice(self.background_noises)
            else:
                noise_path = noise_path_override

                if "noise-sound-bible-0083.wav" in noise_path:
                    noise_dur = 58.61875  # this is the one used in the paper and saved requiring downloading Musan
                else:
                    for noise_path, noise_dur in self.background_noises:
                        if noise_path == noise_path_override:
                            break
            assert Path(noise_path).exists(), f"Invalid noise path: {noise_path}"
            sounds.append(noise_path)
            start_times.append(str(total_noise_dur))
            chosen_snr = snr if snr is not None else quick_choice([15, 10])
            snrs.append(str(chosen_snr))
            # Space with a 1s interval
            total_noise_dur += noise_dur + 1

        start_times = ",".join(start_times)
        snrs = ",".join(snrs)
        sounds = ",".join(sounds)

        background_sound_cmd = self.BACKGROUND_NOISE_CMD.substitute(sounds=sounds, start_times=start_times, snrs=snrs)
        wav_out = self.apply_sox_cmd_to_audio(wav, cmd=background_sound_cmd)
        return wav_out

    def apply_music(self, wav: WAVFile, snr: int = None, music_path: str = None, file_id: int = None) -> WAVFile:
        """
        Apply background music to the audio. We use music from the MUSAN dataset.
        """
        if music_path is None:
            assert self.background_sound_dir is not None, "Background sound directory is not set"

        if file_id is not None:
            assert music_path is None, "Cannot specify both file_id and music_path"
            file_id = file_id % len(self.background_music)
            music_path, _ = self.background_music[file_id]
        elif music_path is None:
            music_path, _ = quick_choice(self.background_music)

        assert Path(music_path).exists(), f"Invalid music path: {music_path}"

        chosen_snr = snr if snr is not None else quick_choice([15, 10, 8, 5])

        background_sound_cmd = self.BACKGROUND_NOISE_CMD.substitute(
            sounds=music_path, start_times="0", snrs=str(chosen_snr)
        )
        wav_out = self.apply_sox_cmd_to_audio(wav, cmd=background_sound_cmd)
        return wav_out

    def apply_speech(self, wav: WAVFile, snr: int = None, speech_path: str = None, file_id: int = None) -> WAVFile:
        """
        Apply background speech to the audio. We use speech from the MUSAN dataset.
        """
        if speech_path is None:
            assert self.background_sound_dir is not None, "Background sound directory is not set"

        if file_id is not None:
            assert speech_path is None, "Cannot specify both file_id and speech_path"
            file_id = file_id % len(self.background_speech)
            speech_path, _ = self.background_speech[file_id]
        elif speech_path is None:
            speech_path, _ = quick_choice(self.background_speech)
        assert Path(speech_path).exists(), f"Invalid speech path: {speech_path}"
        chosen_snr = snr if snr is not None else quick_choice([20, 17])

        background_sound_cmd = self.BACKGROUND_NOISE_CMD.substitute(
            sounds=speech_path, start_times="0", snrs=str(chosen_snr)
        )
        wav_out = self.apply_sox_cmd_to_audio(wav, cmd=background_sound_cmd)
        return wav_out

    def apply_speed(self, wav: WAVFile, speed: float) -> WAVFile:
        """
        Apply speed augmentation to the audio.
        speed_factor > 1.0 speeds up the audio, < 1.0 slows it down.
        """
        # Set fixed seeds for numpy and Python's random module
        speed_cmd = self.SPEED_CMD.substitute(speed=speed)
        wav_out = self.apply_sox_cmd_to_audio(wav, cmd=speed_cmd)
        return wav_out

    def apply_trim(self, wav: WAVFile | str | Path, start: float, end: float) -> WAVFile:
        """
        Apply trim augmentation to the audio.
        """
        if isinstance(wav, str | Path):
            wav = WAVFile(wav)
        trim_cmd = self.TRIM_CMD.substitute(start=start, end=end)
        wav_out = self.apply_sox_cmd_to_audio(wav, cmd=trim_cmd)
        return wav_out

    def apply_colored_noise(self, wav: WAVFile, noise_type: str, snr: float) -> WAVFile:
        """
        Apply color noise to the audio using SoX.

        Args:
            wav (WAVFile): The WAVFile object containing the audio data.
            noise_type (str): Type of noise to apply ("white", "pink", "brown", "blue", "violet", "gray").
            snr (float): Signal-to-noise ratio (lower is more noise).

        Returns:
            WAVFile: The audio with added noise.
        """
        assert noise_type in self.colored_noise_files, f"Invalid noise type: {noise_type}"
        colored_noise_path = self.colored_noise_files[noise_type]
        assert colored_noise_path is not None, f"Noise file for {noise_type} is not set"
        assert Path(colored_noise_path).exists(), f"Noise file not found: {colored_noise_path}"

        # Calculate noise volume based on SNR
        signal_power = np.mean(wav.audio.astype(np.float64) ** 2)
        noise_wav = WAVFile.from_file(colored_noise_path)
        noise_power = np.mean(noise_wav.audio.astype(np.float64) ** 2)
        noise_volume = np.sqrt(signal_power / (10 ** (snr / 10) * noise_power))

        # Prepare the SoX command
        color_noise_cmd = self.COLOR_NOISE_CMD.substitute(
            noise_file=colored_noise_path, noise_volume=noise_volume, duration=len(wav.audio) / TARGET_SAMPLING_RATE
        )

        # Run SoX command using pipes for in-memory processing
        wav_out = self.apply_sox_cmd_to_audio(wav, cmd=color_noise_cmd)

        return wav_out
