from collections.abc import AsyncIterator

from common.app.modules.tts.providers.base import TTSBaseProvider, TTSProviderLiteral
from common.app.modules.tts.providers.smallest_ai import SmallestAITTSProvider


class TTSProvider:
    @staticmethod
    def get_provider(provider: str) -> TTSBaseProvider:
        if provider == "smallest_ai":
            return SmallestAITTSProvider()
        raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    async def generate_audio(
        cls,
        text: str,
        voice: TTSBaseProvider.TTSVoice,
        audio_config: TTSBaseProvider.TTSAudioConfig | None = None,
        provider: TTSProviderLiteral = "smallest_ai",
    ) -> bytes:
        tts_provider = TTSProvider.get_provider(provider)
        if audio_config is None:
            audio_config = tts_provider.TTSAudioConfig()
        return await tts_provider.tts(text, voice, audio_config)

    @classmethod
    async def generate_audio_stream(
        cls,
        text: str,
        voice: TTSBaseProvider.TTSVoice,
        audio_config: TTSBaseProvider.TTSAudioConfig | None = None,
        provider: TTSProviderLiteral = "smallest_ai",
    ) -> AsyncIterator[bytes]:
        tts_provider = TTSProvider.get_provider(provider)
        if audio_config is None:
            audio_config = tts_provider.TTSAudioConfig()
        return tts_provider.tts_stream(text, voice, audio_config)
