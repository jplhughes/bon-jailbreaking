from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import anthropic.types
import numpy as np
import openai.types.chat
import pydantic
from termcolor import cprint
from typing_extensions import Self

from ..utils.audio_utils import get_audio_data, prepare_audio_part, prepare_openai_s2s_audio
from ..utils.image_utils import image_to_base64
from .hashable import HashableBaseModel
from .inference import LLMResponse

PRINT_COLORS = {
    "user": "cyan",
    "system": "magenta",
    "assistant": "light_green",
    "audio": "yellow",
    "image": "yellow",
    "none": "cyan",
}


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"
    audio = "audio"
    image = "image"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class ChatMessage(HashableBaseModel):
    role: MessageRole
    content: str | Path

    def __post_init__(self):
        if isinstance(self.content, Path):
            self.content = str(self.content)

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def openai_format(self) -> openai.types.chat.ChatCompletionMessageParam:
        return {"role": self.role.value, "content": self.content}

    def openai_image_format(self):
        # for images the format involves including images and user text in the same message
        if self.role == MessageRole.image:
            base64_image = image_to_base64(self.content)
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                },
            }
        elif self.role == MessageRole.user:
            return {"type": "text", "text": self.content}
        else:
            raise ValueError(f"Invalid role {self.role} in prompt when using images")

    def anthropic_format(self) -> anthropic.types.MessageParam:
        assert self.role.value in ("user", "assistant")
        return anthropic.types.MessageParam(content=self.content, role=self.role.value)

    def gemini_format(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    def remove_role(self) -> Self:
        return self.__class__(role=MessageRole.none, content=self.content)


class PromptTemplate(pydantic.BaseModel):
    method: str
    messages: Sequence[ChatMessage]
    messages_followup: Sequence[ChatMessage] | None = None
    extra: dict[str, str] = {}


class Prompt(HashableBaseModel):
    messages: Sequence[ChatMessage]

    def __str__(self) -> str:
        out = ""
        for msg in self.messages:
            if msg.role != MessageRole.none:
                out += f"\n\n{msg.role.value}: {msg.content}"
            else:
                out += f"\n{msg.content}"
        return out.strip()

    def __add__(self, other: Self) -> Self:
        return self.__class__(messages=list(self.messages) + list(other.messages))

    @classmethod
    def from_alm_input(
        cls,
        audio_file: str | Path | np.ndarray | None = None,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> Self:
        if audio_file is None and user_prompt is None:
            raise ValueError("Either audio_file or user_prompt must be provided")

        messages = []
        if audio_file != "":
            messages.append(ChatMessage(role=MessageRole.audio, content=audio_file))
        if system_prompt is not None:
            messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
        if user_prompt is not None:
            messages.append(ChatMessage(role=MessageRole.user, content=user_prompt))

        return cls(messages=messages)

    @classmethod
    def from_almj_prompt_format(
        cls,
        text: str,
        sep: str = 8 * "=",
        strip_content: bool = False,
    ) -> Self:
        if not text.startswith(sep):
            return cls(
                messages=[
                    ChatMessage(
                        role=MessageRole.user,
                        content=text.strip() if strip_content else text,
                    )
                ]
            )

        messages = []
        for role_content_str in ("\n" + text).split("\n" + sep):
            if role_content_str == "":
                continue

            role, content = role_content_str.split(sep + "\n")
            if strip_content:
                content = content.strip()

            messages.append(ChatMessage(role=MessageRole[role], content=content))

        return cls(messages=messages)

    def is_none_in_messages(self) -> bool:
        return any(msg.role == MessageRole.none for msg in self.messages)

    def is_last_message_assistant(self) -> bool:
        return self.messages[-1].role == MessageRole.assistant

    def contains_image(self) -> bool:
        return any(msg.role == MessageRole.image for msg in self.messages)

    def add_assistant_message(self, message: str) -> "Prompt":
        return self + Prompt(messages=[ChatMessage(role=MessageRole.assistant, content=message)])

    def add_user_message(self, message: str) -> "Prompt":
        return self + Prompt(messages=[ChatMessage(role=MessageRole.user, content=message)])

    def add_audio_message(self, message: str) -> "Prompt":
        return self + Prompt(messages=[ChatMessage(role=MessageRole.audio, content=message)])

    def hf_format(self, hf_model_id: str) -> str:
        match hf_model_id:
            case "cais/zephyr_7b_r2d2" | "HuggingFaceH4/zephyr-7b-beta":
                # See https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
                # and https://github.com/centerforaisafety/HarmBench/blob/1751dd591e3be4bb52cab4d926977a61e304aba5/baselines/model_utils.py#L124-L127
                # for prompt format.
                rendered_prompt = ""
                for msg in self.messages:
                    match msg.role:
                        case MessageRole.system:
                            rendered_prompt += "<|system|>"
                        case MessageRole.user:
                            rendered_prompt += "<|user|>"
                        case MessageRole.assistant:
                            rendered_prompt += "<|assistant|>"
                        case _:
                            raise ValueError(f"Invalid role {msg.role} in prompt")

                    rendered_prompt += f"\n{msg.content}</s>\n"

                match self.messages[-1].role:
                    case MessageRole.user:
                        rendered_prompt += "<|assistant|>\n"
                    case _:
                        raise ValueError("Last message in prompt must be user. " f"Got {self.messages[-1].role}")

                return rendered_prompt

            case _:
                return "\n\n".join(msg.content for msg in self.messages)

    def openai_format(
        self,
    ) -> list[openai.types.chat.ChatCompletionMessageParam]:
        if self.is_last_message_assistant():
            raise ValueError(
                f"OpenAI chat prompts cannot end with an assistant message. Got {self.messages[-1].role}: {self.messages[-1].content}"
            )
        if self.is_none_in_messages():
            raise ValueError(f"OpenAI chat prompts cannot have a None role. Got {self.messages}")
        if self.contains_image():
            return self.openai_image_format()
        return [msg.openai_format() for msg in self.messages]

    def gemini_format(self, use_vertexai: bool = False) -> List[str]:
        if self.is_none_in_messages():
            raise ValueError(f"Gemini chat prompts cannot have a None role. Got {self.messages}")

        messages = []

        for msg in self.messages:
            if msg.role == MessageRole.audio:
                messages.append(prepare_audio_part(msg.content, use_vertexai))
            elif msg.role == MessageRole.user:
                if msg.content == "" or msg.content is None:
                    messages.append(" ")
                else:
                    messages.append(msg.content)
        return messages

    def openai_s2s_format(self) -> List[Any]:
        messages = []

        for msg in self.messages:
            assert (
                msg.role == MessageRole.audio
            ), f"GPT-4o S2S API does not accept non-audio content! Received a message of type {msg.role}"
            messages.append(prepare_openai_s2s_audio(msg.content))
        return messages

    def openai_image_format(self) -> List[Any]:
        messages = []

        content = []
        for i, msg in enumerate(self.messages):
            if msg.role == MessageRole.system:
                assert i == 0, "System message must be first"
                messages.append(msg.openai_format())
            elif msg.role == MessageRole.image or msg.role == MessageRole.user:
                content.append(msg.openai_image_format())
            else:
                raise ValueError(f"Invalid role {msg.role} in prompt")

        messages.append({"role": "user", "content": content})

        return messages

    def anthropic_format(
        self,
    ) -> tuple[str | None, list[anthropic.types.MessageParam]]:
        """Returns (system_message (optional), chat_messages)"""
        if self.is_none_in_messages():
            raise ValueError(f"Anthropic chat prompts cannot have a None role. Got {self.messages}")

        if len(self.messages) == 0:
            return None, []

        if self.messages[0].role == MessageRole.system:
            return self.messages[0].content, [msg.anthropic_format() for msg in self.messages[1:]]

        return None, [msg.anthropic_format() for msg in self.messages]

    def pretty_print(self, responses: list[LLMResponse], print_fn: Callable | None = None) -> None:
        if print_fn is None:
            print_fn = cprint

        for msg in self.messages:
            if msg.role != MessageRole.none:
                print_fn(f"=={msg.role.upper()}:", "white")
            print_fn(msg.content, PRINT_COLORS[msg.role])
        for i, response in enumerate(responses):
            print_fn(f"==RESPONSE {i + 1} ({response.model_id}):", "white")
            print_fn(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
        print_fn("")


class BatchPrompt(pydantic.BaseModel):
    prompts: Sequence[Prompt]

    @classmethod
    def from_alm_batch_input(
        cls,
        audio_inputs: Union[List[str], List[Path], List[np.ndarray], None] = None,
        user_prompts: List[str] | None = None,
        system_prompts: List[str] | None = None,
    ) -> "BatchPrompt":
        if audio_inputs is None and user_prompts is None:
            raise ValueError("Either audio_inputs or user_prompts must be provided")

        if audio_inputs is not None and user_prompts is not None and len(audio_inputs) != len(user_prompts):
            raise ValueError("audio_inputs and user_prompts must have the same length")

        prompts = []
        for audio, user_prompt, system_prompt in zip(audio_inputs, user_prompts, system_prompts):
            prompts.append(
                Prompt.from_alm_input(audio_file=audio, user_prompt=user_prompt, system_prompt=system_prompt)
            )
        return cls(prompts=prompts)

    def batch_format(self) -> Tuple[np.ndarray, List]:
        audio_messages = []
        text_messages = []
        system_messages = []

        for prompt in self.prompts:
            audio_msg = None
            user_msg = None
            system_msg = None

            for msg in prompt.messages:
                if msg.role == MessageRole.audio:
                    if audio_msg is not None:
                        raise ValueError("Multiple audio messages found in a single prompt")
                    audio_msg = get_audio_data(msg.content)
                elif msg.role == MessageRole.user:
                    if user_msg is not None:
                        raise ValueError("Multiple user messages found in a single prompt")
                    user_msg = msg.content if msg.content and msg.content != "" else " "
                elif msg.role == MessageRole.system:
                    if system_msg is not None:
                        raise ValueError("Multiple system messages found in a single prompt")
                    system_msg = msg.content

            if audio_msg is None:
                raise ValueError("No audio message found in prompt")
            if user_msg is None:
                raise ValueError("No user message found in prompt")

            audio_messages.append(audio_msg)
            text_messages.append(user_msg)
            system_messages.append(system_msg)  # This can be None if no system message was present

        # Now process audio_messages to ensure they're all the same length
        max_length = max(audio.shape[0] for audio in audio_messages)
        padded_audio = []
        for audio in audio_messages:
            pad_length = max_length - audio.shape[0]
            assert audio.ndim == 1, "Audio should only ever have 1 channel"
            padded = np.pad(audio, (0, pad_length), mode="constant")
            padded_audio.append(padded)

        audio_messages = np.stack(padded_audio)

        assert len(audio_messages) == len(
            text_messages
        ), f"Not the same number of text and audio messages! {len(text_messages)} text messages and {len(audio_messages)} audio messages!"
        return audio_messages, text_messages, system_messages

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index) -> Prompt:
        return self.prompts[index]

    def __iter__(self):
        return iter(self.prompts)
