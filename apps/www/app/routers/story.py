import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from apps.www.app.models.api.requests.story import GenerateStoryRequest
from apps.www.app.models.api.responses.story import GenerateStoryResponse
from apps.www.app.services.story import StoryService
from apps.www.core.config import config

router = APIRouter(
    prefix=config.SERVICE_ROUTE_PREFIX + "/story",
)


def _require_shared_llm_clients(request: Request):
    clients = request.app.state.shared_llm_clients
    if clients is None:
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured (OPENROUTER_API_KEY is required).",
        )
    return clients


@router.post("/")
async def generate_story(data: GenerateStoryRequest, request: Request):
    clients = _require_shared_llm_clients(request)
    story_text, url = await StoryService.generate_story(
        data, shared_llm_clients=clients
    )
    return GenerateStoryResponse(story=story_text, audio_url=url)


@router.post("/sse")
async def generate_story_sse(data: GenerateStoryRequest, request: Request):
    clients = _require_shared_llm_clients(request)

    async def event_stream():
        async for event, text in StoryService.generate_story_sse(
            data, shared_llm_clients=clients
        ):
            yield f"data: {json.dumps({'event': event, 'text': text})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")