from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel

from common.core.config import config

TTSProviderLiteral = Literal["smallest_ai"]


class TTSBaseProvider(BaseModel):
    provider: TTSProviderLiteral = "smallest_ai"

    class TTSVoice(BaseModel):
        language_code: str
        voice_id: str
        model: str | None = None
        gender: Literal["male", "female", "neutral"] | None = "male"

    class TTSAudioConfig(BaseModel):
        audio_encoding: Literal["LINEAR16", "MP3", "OGG_OPUS", "MULAW", "ALAW", "PCM"] = "LINEAR16"
        sample_rate: int | None = 24000
        speed: float | None = 1.0
        consistency: float | None = 0.5
        similarity: float | None = 0.0
        enhancement: int | None = 1

    def get_api_key(self) -> str:
        api_key: str | None = None

        if self.provider == "smallest_ai":
            api_key = config.SMALLEST_AI_API_KEY

        if not api_key:
            raise Exception("API key not found")

        return api_key

    async def tts(
        self,
        text: str,
        voice: TTSVoice,
        audio_config: TTSAudioConfig,
    ) -> bytes: ...

    async def tts_stream(
        self,
        text: str,
        voice: TTSVoice,
        audio_config: TTSAudioConfig,
    ) -> AsyncIterator[bytes]:
        yield b""
