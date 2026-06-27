from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import apps.www.core.config  # noqa: F401 — configure `common` with www settings before routers load
import common.core.logging  # noqa: F401 — Logfire + root logging once env is configured
from apps.www.app.routers.story import router as story_router
from apps.www.app.routers.system import router as system_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from common.app.modules.llm.clients import SharedLLMClients, create_shared_llm_clients
    from common.core.db.mongo import MongoApp, create_mongo_app

    clients: SharedLLMClients | None = None
    try:
        clients = create_shared_llm_clients()
    except ValueError:
        clients = None
    app.state.shared_llm_clients = clients

    mongo: MongoApp | None = create_mongo_app()
    if mongo is not None:
        await mongo.ping()
    app.state.mongo = mongo

    yield
    if clients is not None:
        await clients.aclose()
    if mongo is not None:
        await mongo.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Credentials cannot be combined with a wildcard origin (browsers reject it and the
    # spec disallows it). Keep this False until origins are an explicit allowlist.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(story_router)
