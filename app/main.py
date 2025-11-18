from fastapi import FastAPI

from .config import get_settings
from .routers import ui, api


settings = get_settings()
app = FastAPI(title=settings.app_name)
app.include_router(ui.router)
app.include_router(api.router)


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
