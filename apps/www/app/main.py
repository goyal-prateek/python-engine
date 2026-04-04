import apps.www.core.config  # noqa: F401 — configure `common` with www settings before routers load

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.www.app.routers.story import router as story_router
from apps.www.app.routers.system import router as system_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from common.app.modules.llm.clients import SharedLLMClients, create_shared_llm_clients

    clients: SharedLLMClients | None = None
    try:
        clients = create_shared_llm_clients()
    except ValueError:
        clients = None
    app.state.shared_llm_clients = clients
    yield
    if clients is not None:
        await clients.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(story_router)
