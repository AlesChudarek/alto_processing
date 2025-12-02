from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .routers import ui, api
from .middleware.auth import TokenAuthMiddleware


settings = get_settings()
app = FastAPI(title=settings.app_name)
app.add_middleware(TokenAuthMiddleware, settings=settings)
app.include_router(ui.router)
app.include_router(api.router)

static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
