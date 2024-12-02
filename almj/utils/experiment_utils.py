import dataclasses
import logging
import random
from pathlib import Path

import jsonlines
import numpy as np

from almj.apis import InferenceAPI
from almj.utils import utils

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class ExperimentConfigBase:
    # Experiments with the same output_dir will share the same cache, log
    # directory, and api cost tracking.
    output_dir: Path

    # If None, defaults to output_dir / "cache"
    enable_cache: bool = True
    cache_dir: Path | None = None

    # If None, defaults to output_dir / "prompt-history"
    enable_prompt_history: bool = False
    prompt_history_dir: Path | None = None

    log_to_file: bool = True
    openai_fraction_rate_limit: float = 0.5
    openai_num_threads: int = 100
    gemini_num_threads: int = 100
    openai_s2s_num_threads: int = 40
    gpt4o_s2s_rpm_cap: int = 10
    gemini_recitation_rate_check_volume: int = 100
    gemini_recitation_rate_threshold: float = 0.5
    anthropic_num_threads: int = 40
    gray_swan_num_threads: int = 40
    huggingface_num_threads: int = 100
    openai_tag: str = "OPENAI_API_KEY1"
    print_prompt_and_response: bool = False

    datetime_str: str = dataclasses.field(init=False)
    seed: int = 42

    def __post_init__(self):
        self.datetime_str = utils.get_datetime_str()

    def reset_api(self):
        if hasattr(self, "_api"):
            del self._api

    @property
    def api(self) -> InferenceAPI:
        if not hasattr(self, "_api"):
            if self.cache_dir is None:
                self.cache_dir = self.output_dir / "cache"
            if not self.enable_cache:
                self.cache_dir = None
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            if self.prompt_history_dir is None:
                self.prompt_history_dir = self.output_dir / "prompt-history"
            if not self.enable_prompt_history:
                self.prompt_history_dir = None
            if self.prompt_history_dir is not None:
                self.prompt_history_dir.mkdir(parents=True, exist_ok=True)

            self._api = InferenceAPI(
                openai_fraction_rate_limit=self.openai_fraction_rate_limit,
                openai_num_threads=self.openai_num_threads,
                gemini_num_threads=self.gemini_num_threads,
                openai_s2s_num_threads=self.openai_s2s_num_threads,
                gpt4o_s2s_rpm_cap=self.gpt4o_s2s_rpm_cap,
                gemini_recitation_rate_check_volume=self.gemini_recitation_rate_check_volume,
                gemini_recitation_rate_threshold=self.gemini_recitation_rate_threshold,
                anthropic_num_threads=self.anthropic_num_threads,
                gray_swan_num_threads=self.gray_swan_num_threads,
                huggingface_num_threads=self.huggingface_num_threads,
                cache_dir=self.cache_dir,
                prompt_history_dir=self.prompt_history_dir,
            )

        return self._api

    def setup_experiment(self, log_file_prefix: str = "run") -> None:
        """Should only be called once at the beginning of the experiment."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # set seed
        random.seed(self.seed)
        np.random.seed(42)

        # Set up logging
        if self.log_to_file:
            (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
            logfile = self.output_dir / "logs" / f"{log_file_prefix}-{self.datetime_str}.log"
            print(f"Logging to {logfile}")
            print(f"You can tail the log file with: tail -f {logfile.resolve()}")
        else:
            logfile = None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=logfile,
        )
        # Log the experiment configuration
        LOGGER.info(f"Experiment configuration: {self}")

        # Do remaining environment setup after we set up our logging, so that
        # our basicConfig is the one that gets used.
        utils.setup_environment(openai_tag=self.openai_tag)

    def log_api_cost(self, metadata: dict[str, str] | None = None):
        if metadata is None:
            metadata = {}
        if not hasattr(self, "_last_api_cost"):
            self._last_api_cost = 0

        api_cost_file = self.output_dir / "api-costs.jsonl"
        with jsonlines.open(api_cost_file, mode="a") as writer:
            writer.write({"cost": self.api.running_cost - self._last_api_cost} | metadata)
        self._last_api_cost = self.api.running_cost
