from fastapi import APIRouter

from apps.www.core.config import config

router = APIRouter(
    prefix=config.SERVICE_ROUTE_PREFIX + "/system",
)


@router.get("/health/")
async def health():
    return {"status": "ok"}
