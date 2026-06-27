from fastapi import APIRouter, HTTPException, Request

from apps.www.core.config import config
from common.app.modules.db.mongo import MongoPersistentStoreHealth
from common.core.db.mongo import MongoApp

router = APIRouter(
    prefix=config.SERVICE_ROUTE_PREFIX + "/system",
)


@router.get("/health/")
async def health(request: Request):
    mongo: MongoApp | None = getattr(request.app.state, "mongo", None)
    if mongo is None:
        return {"status": "ok", "mongo": "disabled"}

    try:
        await MongoPersistentStoreHealth(mongo).check_reachable()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "mongo": "unreachable", "error": str(exc)},
        ) from exc

    return {"status": "ok", "mongo": "ok"}
