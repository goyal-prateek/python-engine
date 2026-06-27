from typing import Literal

from pydantic import BaseModel


class LLMPromptItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str
