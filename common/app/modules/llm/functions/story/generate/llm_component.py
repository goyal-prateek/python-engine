from typing import Any

from common.app.modules.llm.functions import LLMFunction
from common.app.modules.llm.functions.story.models import (
    GenerateStoryOutputModel,
    GenerateStoryParamsModel,
)
from common.app.modules.llm.promtps import LLMPromptItem


def generate_story_transfrom_params(params: GenerateStoryParamsModel) -> dict[str, Any]:
    return {}


def extend_prompt(
    step_input: dict[str, Any], params: LLMFunction.ParamsModel
) -> list[LLMPromptItem]:
    assert isinstance(params, GenerateStoryParamsModel)

    return [
        LLMPromptItem(
            role="user",
            content=f"""
    Generate a story about {params.theme} in {params.language} with {params.genre} genre.
    The story should be around 200 to 300 words long.
    The story should be {params.story_idea} story idea.
    The story should be {params.tone} tone.
    """,
        ),
    ]


generate_story_llm_component = LLMFunction(
    name="generate_story",
    params_model=GenerateStoryParamsModel,
    output_model=GenerateStoryOutputModel,
    steps=[
        LLMFunction.TransformStep(function=generate_story_transfrom_params),
        LLMFunction.CompletionStep(
            model="google/gemini-2.0-flash-exp:free",
            extend_prompt=extend_prompt,
        ),
    ],
    stream=True,
)
