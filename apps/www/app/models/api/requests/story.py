from typing import Literal

from pydantic import BaseModel


class GenerateStoryRequest(BaseModel):
    genre: str
    theme: str
    language: Literal["english", "hindi"] | None = "english"
    target_length: Literal["1000_words", "2500_words", "4000_words"] | None = "1000_words"
    story_idea: str | None = None
    tone: str | None = None
