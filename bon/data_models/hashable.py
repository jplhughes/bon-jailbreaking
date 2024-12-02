import hashlib

import pydantic


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


class HashableBaseModel(pydantic.BaseModel):
    def model_hash(self) -> str:
        as_json = self.model_dump_json()
        return deterministic_hash(as_json)

    model_config = pydantic.ConfigDict(frozen=True)
