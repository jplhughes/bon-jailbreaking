import json
import os
import re
from pathlib import Path
from typing import Dict, List

import requests


def load_voice_dict():
    # Construct the path to the voice_dict.json file using pathlib
    voices_file_path = Path(__file__).resolve().parent / "voice_dict.json"
    with open(voices_file_path, "r") as file:
        voice_dict = json.load(file)

    return voice_dict, voices_file_path


def query_voices() -> List[Dict]:
    url = "https://api.elevenlabs.io/v1/voices"

    headers = {
        "Accept": "application/json",
        "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    return data["voices"]


def update_voice_dict(pulled_voices: List[Dict]):
    """
    Updates the voice_dict.json file with the latest voices from the ElevenLabs API
    This is then used to help populate voices.py
    """
    voice_dict, voices_file_path = load_voice_dict()
    current_ids = [v.get("id") for v in voice_dict.values()]

    for voice in pulled_voices:
        # If voice id is already in our VOICE_DICT we don't want to re-add it or overwrite
        if voice["voice_id"] in current_ids:
            continue

        # If its not then we want to add
        else:
            voice_dict[voice["name"]] = {
                "id": voice["voice_id"],
                "gender": voice["labels"]["gender"],
                "age": voice["labels"]["age"],
                "nationality": voice["labels"]["accent"],
                "voice": (
                    voice["labels"]["description"]
                    if "description" in voice["labels"]
                    else voice["labels"]["descriptive"]
                ),
                "use": voice["labels"]["use_case"] if "use_case" in voice["labels"] else None,
                "description": voice["description"],
            }

    with open(voices_file_path, "w") as f:
        json.dump(voice_dict, f)


def split_text(text, max_length=400):
    sentences = re.split("(?<=[.!?]) +", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

        # Check if the current sentence alone exceeds max_length
        if len(sentence) > max_length:
            # Split the sentence into smaller chunks
            while len(sentence) > max_length:
                chunks.append(sentence[:max_length])
                sentence = sentence[max_length:]
            if sentence:
                current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Ensure no chunk exceeds max_length
    return [chunk[:max_length] for chunk in chunks]
