from typing import AsyncGenerator, Literal, Tuple
import uuid

from apps.www.app.models.api.requests.story import GenerateStoryRequest
from common.app.modules.llm.clients import SharedLLMClients
from common.app.modules.llm.functions.story.generate.llm_component import (
    generate_story_llm_component,
)
from common.app.modules.llm.functions.story.models import (
    GenerateStoryOutputModel,
    GenerateStoryParamsModel,
)
from common.app.modules.tts.providers import TTSProvider
from common.app.modules.tts.providers.base import TTSBaseProvider
from common.app.modules.tts.providers.smallest_ai import SmallestAITTSProvider
from common.core.s3 import async_s3_client


class StoryService:
    @classmethod
    async def generate_story(
        cls,
        data: GenerateStoryRequest,
        *,
        shared_llm_clients: SharedLLMClients,
    ) -> Tuple[str, str]:
        story_text = ""
        llm_execution = await generate_story_llm_component.run(
            params=GenerateStoryParamsModel(
                theme=data.theme,
                language=data.language,
                genre=data.genre,
                target_length=data.target_length,
                story_idea=data.story_idea,
                tone=data.tone,
            ),
            shared_llm_clients=shared_llm_clients,
        )
        assert isinstance(llm_execution, AsyncGenerator)
        async for chunk in llm_execution:
            story_text += chunk

        if data.language == "hindi":
            voice_name = "irisha"
            voice_language_code: Literal["en", "hi", "ta", "fr", "de", "pl"] = "hi"
        else:
            voice_name = "emily"
            voice_language_code = "en"

        audio_content = await TTSProvider.generate_audio(
            text=story_text,
            voice=SmallestAITTSProvider.TTSVoice(
                language_code=voice_language_code,
                voice_id=voice_name,
                model="lightning",
            ),
            audio_config=TTSBaseProvider.TTSAudioConfig(),
            provider="smallest_ai",
        )

        key = f"stories/{uuid.uuid4()}.wav"

        await async_s3_client.put_object(
            Bucket="taletalk",
            Key=key,
            Body=audio_content,
        )

        url = await async_s3_client.get_s3_url(Bucket="taletalk", Key=key)

        return (story_text, url)

    @classmethod
    async def generate_story_sse(
        cls,
        data: GenerateStoryRequest,
        *,
        shared_llm_clients: SharedLLMClients,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        story_text = ""
        llm_execution = await generate_story_llm_component.run(
            params=GenerateStoryParamsModel(
                theme=data.theme,
                language=data.language,
                genre=data.genre,
                target_length=data.target_length,
                story_idea=data.story_idea,
                tone=data.tone,
            ),
            shared_llm_clients=shared_llm_clients,
        )
        assert isinstance(llm_execution, AsyncGenerator)
        line_to_send, generated_story = "", ""
        async for chunk in llm_execution:
            story_text += chunk
            generated_story += chunk
            if "\n" in generated_story:
                line_to_send, generated_story = generated_story.split("\n", 1)
                yield ("story_text_append", line_to_send)
            
        for line_to_send in generated_story.split("\n"):
            yield ("story_text_append", line_to_send)

        if data.language == "hindi":
            voice_name = "irisha"
            voice_language_code: Literal["en", "hi", "ta", "fr", "de", "pl"] = "hi"
        else:
            voice_name = "emily"
            voice_language_code = "en"

        audio_content = await TTSProvider.generate_audio(
            text=story_text,
            voice=SmallestAITTSProvider.TTSVoice(
                language_code=voice_language_code,
                voice_id=voice_name,
                model="lightning",
            ),
            audio_config=TTSBaseProvider.TTSAudioConfig(),
            provider="smallest_ai",
        )

        key = f"stories/{uuid.uuid4()}.wav"

        await async_s3_client.put_object(
            Bucket="taletalk",
            Key=key,
            Body=audio_content,
        )

        url = await async_s3_client.get_s3_url(Bucket="taletalk", Key=key)

        yield ("audio_url", url)