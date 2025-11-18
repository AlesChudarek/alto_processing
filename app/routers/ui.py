from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..dependencies import get_app_settings, get_templates
from ..core.comparison_legacy import MODEL_REGISTRY_JSON, DEFAULT_AGENT_PROMPT_TEXT, DEFAULT_MODEL

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def show_form(
    request: Request,
    templates=Depends(get_templates),
    settings=Depends(get_app_settings),
) -> HTMLResponse:
    return templates.TemplateResponse(
        "compare.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "model_registry_json": MODEL_REGISTRY_JSON,
            "default_agent_model": DEFAULT_MODEL,
            "default_agent_prompt": DEFAULT_AGENT_PROMPT_TEXT,
        },
    )
