import uvicorn

from apps.www.core.config import config

if __name__ == "__main__":
    uvicorn.run(
        app="apps.www.app.main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=config.HOT_RELOAD,
    )

