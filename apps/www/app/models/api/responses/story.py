from typing import Literal

from pydantic import BaseModel


class GenerateStoryResponse(BaseModel):
    story: str
    audio_url: str
    success: Literal["ok"] = "ok"
