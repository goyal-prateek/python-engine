from typing import Literal

from common.app.modules.llm.functions import LLMFunction


class GenerateStoryParamsModel(LLMFunction.ParamsModel):
    genre: str
    theme: str
    language: Literal["english", "hindi"] | None = "english"
    target_length: Literal["1000_words", "2500_words", "4000_words"] | None = "1000_words"
    story_idea: str | None = None
    tone: str | None = None


class GenerateStoryOutputModel(LLMFunction.OutputModel):
    story: str
