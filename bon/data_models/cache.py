import pydantic

from .inference import LLMParams, LLMResponse, TaggedModeration
from .messages import Prompt


class LLMCache(pydantic.BaseModel):
    params: LLMParams
    prompt: Prompt
    responses: list[LLMResponse] | None = None


class LLMCacheModeration(pydantic.BaseModel):
    texts: list[str]
    moderation: list[TaggedModeration] | None = None
