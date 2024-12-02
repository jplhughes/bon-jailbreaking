import base64

import numpy as np
import pydantic

from .hashable import HashableBaseModel


class EmbeddingParams(HashableBaseModel):
    model_id: str
    texts: list[str]
    dimensions: int | None = None

    model_config = pydantic.ConfigDict(extra="forbid", protected_namespaces=())


class EmbeddingResponseBase64(pydantic.BaseModel):
    model_id: str
    embeddings: list[str]

    tokens: int
    cost: float

    model_config = pydantic.ConfigDict(extra="forbid", protected_namespaces=())

    def get_numpy_embeddings(self):
        return np.stack(
            [
                np.frombuffer(
                    buffer=base64.b64decode(x),
                    dtype="float32",
                )
                for x in self.embeddings
            ]
        )
