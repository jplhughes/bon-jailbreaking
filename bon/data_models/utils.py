import asyncio
import logging
import time
from collections import deque
from enum import Enum
from functools import partial
from typing import List

import google.generativeai as genai

# from google.generativeai.types import HarmCategory, HarmProbability
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

GEMINI_MODELS = {
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
}

GEMINI_RPM = {"gemini-1.5-flash-001": 1000, "gemini-1.5-pro-001": 360}
GEMINI_TPM = 2000000

DELETE_FILE_QUOTA = 500


class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    def __init__(self, refresh_rate, total=0, throughput=0):
        self.refresh_rate = refresh_rate
        self.total = total
        self.throughput = throughput
        self.last_update_time = time.time()
        self.start_time = time.time()
        self.value = self.refresh_rate

    def _replenish(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(
            self.refresh_rate,
            self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60,
        )
        self.last_update_time = curr_time
        self.throughput = self.total / (curr_time - self.start_time) * 60

    def geq(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(amount), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount
        self.total += amount


class GeminiRateTracker:
    def __init__(self, rpm_limit, tpm_limit, window_size=60):
        self.window_size = window_size  # Window size in seconds
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.requests = deque()
        self.tokens = deque()
        self.total_requests = 0
        self.total_tokens = 0

    def add_request(self, tokens):
        current_time = time.time()
        self.requests.append(current_time)
        self.tokens.append((current_time, tokens))
        self.total_requests += 1
        self.total_tokens += tokens

        # Remove requests older than the window size
        while self.requests and current_time - self.requests[0] >= self.window_size:
            self.requests.popleft()
            self.total_requests -= 1

    def get_token_count(self):
        self.cleanup()
        return self.total_tokens

    def get_request_count(self):
        self.cleanup()
        return self.total_requests

    def cleanup(self):
        current_time = time.time()
        while self.requests and current_time - self.requests[0] >= self.window_size:
            self.requests.popleft()
            self.total_requests -= 1

        while self.tokens and current_time - self.tokens[0][0] >= self.window_size:
            _, tokens = self.tokens.popleft()
            self.total_tokens -= tokens

    def can_make_request(self, new_tokens):
        self.cleanup()
        return (self.total_requests < self.rpm_limit) and (self.total_tokens + new_tokens <= self.tpm_limit)

    def __str__(self):
        return f"Requests in the last minute: {self.get_request_count()}, Tokens in the last minute: {self.get_token_count()}"


class RecitationRateFailureError(Exception):
    pass


class HarmCategory(str, Enum):
    HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
    HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"


HARM_CATEGORIES_MAP = {
    7: HarmCategory.HARASSMENT,
    8: HarmCategory.HATE_SPEECH,
    9: HarmCategory.SEXUALLY_EXPLICIT,
    10: HarmCategory.DANGEROUS_CONTENT,
}

# We have to have this other mapping because, inexplicably VertexAI and Google.GenerativeAI have different numeric values for their
# harm categories
VERTEXAI_HARM_CATEGORIES_MAP = {
    1: HarmCategory.HATE_SPEECH,
    2: HarmCategory.DANGEROUS_CONTENT,
    3: HarmCategory.HARASSMENT,
    4: HarmCategory.SEXUALLY_EXPLICIT,
}


class HarmProbability(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


HARM_PROBABILITY_MAP = {
    1: HarmProbability.NEGLIGIBLE,
    2: HarmProbability.LOW,
    3: HarmProbability.MEDIUM,
    4: HarmProbability.HIGH,
}


class HarmSeverity(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


HARM_SEVERITY_MAP = {
    1: HarmSeverity.NEGLIGIBLE,
    2: HarmSeverity.LOW,
    3: HarmSeverity.MEDIUM,
    4: HarmSeverity.HIGH,
}


class SafetyRating(BaseModel):
    category: HarmCategory
    probability: HarmProbability
    probabilityScore: float | None = None
    severity: HarmSeverity | None = None
    severityScore: float | None = None

    def __str__(self):
        return self.value


class SafetyRatings(BaseModel):
    ratings: List[SafetyRating]

    def __str__(self):
        return self.value

    def to_dict(self):
        return {"ratings": [rating.dict() for rating in self.ratings]}


def parse_safety_ratings(safety_ratings):
    ratings = []
    for rating in safety_ratings:
        if hasattr(rating, "severity"):
            ratings.append(
                SafetyRating(
                    category=VERTEXAI_HARM_CATEGORIES_MAP[rating.category.value],
                    probability=HARM_PROBABILITY_MAP[rating.probability.value],
                    probabilityScore=rating.probability_score,
                    severity=HARM_SEVERITY_MAP[rating.severity.value],
                    severityScore=rating.severity_score,
                )  # .to_dict()
            )
        else:
            ratings.append(
                SafetyRating(
                    category=HARM_CATEGORIES_MAP[rating.category], probability=HARM_PROBABILITY_MAP[rating.probability]
                )  # .to_dict()
            )
    return SafetyRatings(ratings=ratings).to_dict()


class GeminiStopReason(str, Enum):
    FINISH_REASON_UNSPECIFIED = "finish_reason_unspecified"
    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    SAFETY = "safety"
    RECITATION = "recitation"
    OTHER = "other"
    BLOCKED = "blocked"

    def __str__(self):
        return self.value


class GeminiBlockReason(str, Enum):
    BLOCK_REASON_UNSPECIFIED = "unspecified"
    SAFETY = "safety"
    OTHER = "other"

    def __str__(self):
        return self.value


STOP_REASON_MAP = {
    0: GeminiStopReason.FINISH_REASON_UNSPECIFIED,
    1: GeminiStopReason.STOP,
    2: GeminiStopReason.MAX_TOKENS,
    3: GeminiStopReason.SAFETY,
    4: GeminiStopReason.RECITATION,
    5: GeminiStopReason.OTHER,
}


def get_stop_reason(reason_code):
    return STOP_REASON_MAP[reason_code]


BLOCK_REASON_MAP = {
    0: GeminiBlockReason.BLOCK_REASON_UNSPECIFIED,
    1: GeminiBlockReason.SAFETY,
    2: GeminiBlockReason.OTHER,
}


def get_block_reason(reason_code):
    return BLOCK_REASON_MAP[reason_code]


# Define a function to delete a file (not async)
def delete_genai_file(file):
    try:
        genai.delete_file(file)
        print(f"Successfully deleted file: {file}")
    except Exception as e:
        print(f"Failed to delete file {file}: {e}")


# Define the main function to handle multiple deletions
async def async_delete_genai_files(file_list):
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(100)

    async def sem_delete_file(file):
        async with semaphore:
            await loop.run_in_executor(None, partial(delete_genai_file, file))

    # Create a list of tasks for each file deletion
    tasks = [sem_delete_file(file) for file in file_list]
    # Run the tasks concurrently
    await asyncio.gather(*tasks)
