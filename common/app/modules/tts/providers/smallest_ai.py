from collections.abc import AsyncIterator
from typing import Literal

import aiohttp
from smallestai.waves.utils import add_wav_header, chunk_text

from common.app.modules.tts.providers.base import TTSBaseProvider


class SmallestAITTSProvider(TTSBaseProvider):
    provider: Literal["smallest_ai"] = "smallest_ai"

    class TTSVoice(TTSBaseProvider.TTSVoice):
        model: Literal["lightning", "lightning-large"] = "lightning"
        language_code: Literal["en", "hi", "ta", "fr", "de", "pl"] = "en"

    # async def tts(
    #     self,
    #     text: str,
    #     voice: TTSVoice,
    #     audio_config: TTSBaseProvider.TTSAudioConfig,
    # ) -> bytes:
    #     waves_client = AsyncWavesClient(api_key=self.get_api_key())
    #     audio_content = await waves_client.synthesize(
    #         text=text,
    #     )
    #     # If stream is True, then audio_content is an iterator
    #     # If save_as is provided, then audio_content is None
    #     # Otherwise, audio_content is bytes
    #     assert isinstance(audio_content, bytes)
    #     return audio_content

    async def tts_stream(
        self,
        text: str,
        voice: TTSVoice,
        audio_config: TTSBaseProvider.TTSAudioConfig,
    ) -> AsyncIterator[bytes]:
        async with aiohttp.ClientSession() as session:
            url = f"https://waves-api.smallest.ai/api/v1/{voice.model}/get_speech"
            chunk_size = 250
            if voice.model == "lightning-large":
                chunk_size = 140
            chunks = chunk_text(text, chunk_size)

            for chunk in chunks:
                payload = {
                    "text": chunk,
                    "sample_rate": audio_config.sample_rate,
                    "voice_id": voice.voice_id,
                    "add_wav_header": False,
                    "speed": audio_config.speed,
                    "model": voice.model,
                }

                if voice.model == "lightning-large":
                    if audio_config.consistency is not None:
                        payload["consistency"] = audio_config.consistency
                    if audio_config.similarity is not None:
                        payload["similarity"] = audio_config.similarity
                    if audio_config.enhancement is not None:
                        payload["enhancement"] = audio_config.enhancement

                headers = {
                    "Authorization": f"Bearer {self.get_api_key()}",
                    "Content-Type": "application/json",
                }

                async with session.post(url=url, json=payload, headers=headers) as res:
                    if res.status != 200:
                        raise Exception(
                            f"Failed to synthesize speech: {await res.text()}. For more information, visit https://waves.smallest.ai/"
                        )

                    yield await res.read()

    async def tts(
        self,
        text: str,
        voice: TTSVoice,
        audio_config: TTSBaseProvider.TTSAudioConfig,
    ) -> bytes:
        audio_content = bytearray()
        async for chunk in self.tts_stream(text, voice, audio_config):
            audio_content.extend(chunk)

        sample_rate = audio_config.sample_rate or 24000

        return add_wav_header(bytes(audio_content), sample_rate=sample_rate)
