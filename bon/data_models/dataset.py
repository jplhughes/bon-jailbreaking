import pydantic


class DatasetQuestion(pydantic.BaseModel):
    question_id: int
    question: str
    incorrect_answers: list[str]
    correct_answer: str
