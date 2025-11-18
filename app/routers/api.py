from __future__ import annotations

from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, Body, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from xml.dom import minidom

from ..core.agent_runner import AgentRunnerError, run_agent as run_agent_via_responses
from ..core.comparison_legacy import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    build_agent_diff,
    build_html_diff,
    delete_agent_file,
    describe_library,
    list_agents_files,
    normalize_agent_collection,
    read_agent_file,
    simulate_typescript_processing,
    write_agent_file,
)
from ..core.main_processor import AltoProcessor, DEFAULT_API_BASES

router = APIRouter()


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


@router.get("/process")
def process_page(uuid: str = Query(...), api_base: Optional[str] = Query(None)) -> Response:
    if not uuid:
        return _json_error("UUID je povinný", status_code=400)

    try:
        processor = AltoProcessor(api_base_url=api_base)
        context = processor.get_book_context(uuid)
        if not context:
            return _json_error("Nepodařilo se načíst metadata pro zadané UUID", status_code=404)

        page_uuid = context.get("page_uuid")
        if not page_uuid:
            return _json_error("Nepodařilo se určit konkrétní stránku pro zadané UUID", status_code=404)

        active_api_base = context.get("api_base") or processor.api_base_url
        library_info = describe_library(active_api_base)
        handle_base = library_info.get("handle_base") or ""
        book_uuid = context.get("book_uuid")

        alto_xml = processor.get_alto_data(page_uuid)
        if not alto_xml:
            return _json_error("Nepodařilo se stáhnout ALTO data", status_code=502)

        pretty_alto = minidom.parseString(alto_xml).toprettyxml(indent="  ")
        python_result = processor.get_formatted_text(alto_xml, page_uuid, DEFAULT_WIDTH, DEFAULT_HEIGHT)
        typescript_result = simulate_typescript_processing(alto_xml, page_uuid, DEFAULT_WIDTH, DEFAULT_HEIGHT)

        pages = context.get("pages") or []
        current_index = context.get("current_index", -1)
        total_pages = len(pages)
        has_prev = current_index > 0
        has_next = 0 <= current_index < total_pages - 1
        prev_uuid = pages[current_index - 1]["uuid"] if has_prev else None
        next_uuid = pages[current_index + 1]["uuid"] if has_next else None

        book_data = context.get("book") or {}
        mods_metadata = context.get("mods") or []

        def clean(value: Optional[str]) -> str:
            if not value:
                return ""
            return " ".join(value.replace("\xa0", " ").split())

        page_summary = context.get("page") or {}
        page_item = context.get("page_item") or {}
        book_handle = f"{handle_base}/handle/uuid:{book_uuid}" if handle_base and book_uuid else ""
        page_handle = f"{handle_base}/handle/uuid:{page_uuid}" if handle_base and page_uuid else ""
        details = page_item.get("details") or {}

        page_info = {
            "uuid": page_uuid,
            "title": clean(page_summary.get("title") or page_item.get("title")),
            "pageNumber": clean(page_summary.get("pageNumber") or details.get("pagenumber")),
            "pageType": clean(page_summary.get("pageType") or details.get("type")),
            "pageSide": clean(
                page_summary.get("pageSide")
                or details.get("pageposition")
                or details.get("pagePosition")
                or details.get("pagerole")
            ),
            "index": current_index,
            "iiif": page_item.get("iiif"),
            "handle": page_handle,
            "library": library_info,
        }

        book_info = {
            "uuid": book_uuid,
            "title": clean(book_data.get("title")),
            "model": book_data.get("model"),
            "handle": book_handle,
            "mods": mods_metadata,
            "constants": context.get("book_constants") or {},
            "library": library_info,
        }

        navigation = {
            "hasPrev": has_prev,
            "hasNext": has_next,
            "prevUuid": prev_uuid,
            "nextUuid": next_uuid,
            "total": total_pages,
        }

        response_data = {
            "python": python_result,
            "typescript": typescript_result,
            "book": book_info,
            "pages": pages,
            "currentPage": page_info,
            "navigation": navigation,
            "alto_xml": pretty_alto,
            "library": library_info,
        }
        return JSONResponse(response_data)
    except Exception as exc:
        return _json_error(str(exc), status_code=500)


@router.get("/preview")
def preview_image(uuid: str = Query(...), stream: str = Query("IMG_PREVIEW")) -> Response:
    if not uuid:
        return _json_error("UUID je povinný", status_code=400)

    allowed_streams = {"IMG_THUMB", "IMG_PREVIEW", "IMG_FULL", "AUTO"}
    if stream not in allowed_streams:
        stream = "IMG_PREVIEW"

    candidate_streams = [stream]
    if stream == "AUTO":
        candidate_streams = ["IMG_FULL", "IMG_PREVIEW", "IMG_THUMB"]

    candidate_bases = list(dict.fromkeys(DEFAULT_API_BASES))
    last_error = None

    for candidate in candidate_streams:
        for base in candidate_bases:
            upstream_url = f"{base}/item/uuid:{uuid}/streams/{candidate}"
            try:
                response = requests.get(upstream_url, timeout=20)
            except Exception as exc:
                last_error = str(exc)
                continue

            if response.status_code != 200 or not response.content:
                last_error = f"Nepodařilo se načíst náhled (status {response.status_code} pro {candidate} z {base})"
                response.close()
                continue

            content_type = response.headers.get("Content-Type", "image/jpeg")
            if "jp2" in content_type.lower():
                last_error = f"Stream {candidate} vrací nepodporovaný formát {content_type}"
                response.close()
                continue

            headers = {
                "Cache-Control": "no-store",
                "Content-Length": str(len(response.content)),
                "X-Preview-Stream": candidate,
            }
            data = response.content
            response.close()
            return Response(content=data, media_type=content_type, headers=headers)

    message = last_error or "Nepodařilo se načíst náhled."
    return _json_error(message, status_code=502)


@router.get("/agents")
@router.get("/agents/list")
def list_agents(collection: Optional[str] = Query(None)) -> Dict[str, Any]:
    items = list_agents_files(collection or "")
    return {"agents": items}


@router.get("/agents/get")
def get_agent(name: str = Query(...), collection: Optional[str] = Query(None)) -> Response:
    agent = read_agent_file(name, collection)
    if agent is None:
        return _json_error("not_found", status_code=404)
    return JSONResponse(agent)


@router.post("/diff")
def diff(payload: Dict[str, Any] = Body(...)) -> Response:
    python_html = str(payload.get("python") or payload.get("python_html") or "")
    ts_html = str(payload.get("typescript") or payload.get("typescript_html") or "")
    mode = payload.get("mode")
    if not isinstance(mode, str):
        mode = "word"
    try:
        diff_result = build_html_diff(python_html, ts_html, mode)
        return JSONResponse({"ok": True, "diff": diff_result})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


@router.post("/agents/diff")
def agent_diff(payload: Dict[str, Any] = Body(...)) -> Response:
    original_html = str(payload.get("original") or payload.get("original_html") or "")
    corrected_html = str(payload.get("corrected") or payload.get("corrected_html") or "")
    mode = payload.get("mode")
    if not isinstance(mode, str):
        mode = "word"
    try:
        diff_result = build_agent_diff(original_html, corrected_html, mode)
        return JSONResponse({"ok": True, "diff": diff_result})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


@router.post("/agents/save")
def save_agent(payload: Dict[str, Any] = Body(...)) -> Response:
    collection = payload.get("collection")
    stored = write_agent_file(payload if isinstance(payload, dict) else {}, collection)
    if not stored:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid"})
    data = read_agent_file(stored, collection) or {}
    normalized_collection = normalize_agent_collection(collection)
    body = {
        "ok": True,
        "stored_name": stored,
        "collection": normalized_collection,
        "agent": data,
    }
    return JSONResponse(body)


@router.post("/agents/delete")
def delete_agent(payload: Dict[str, Any] = Body(...)) -> Response:
    name = payload.get("name") if isinstance(payload, dict) else None
    collection = payload.get("collection") if isinstance(payload, dict) else None
    ok = delete_agent_file(name or "", collection)
    if ok:
        return JSONResponse({"ok": True})
    return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})


@router.post("/agents/run")
def run_agent(payload: Dict[str, Any] = Body(...)) -> Response:
    request_payload = payload if isinstance(payload, dict) else {}
    collection = request_payload.get("collection")
    agent_name = request_payload.get("name")
    agent = read_agent_file(agent_name or "", collection)
    if not agent:
        return JSONResponse(status_code=404, content={"ok": False, "error": "agent_not_found"})

    model_override = str(request_payload.get("model_override") or request_payload.get("model") or "").strip()
    reasoning_override = str(request_payload.get("reasoning_effort") or "").strip().lower()
    snapshot = request_payload.get("agent_snapshot")

    if isinstance(snapshot, dict):
        agent_for_run = dict(agent)
        agent_for_run.update(snapshot)
    else:
        agent_for_run = dict(agent)

    agent_for_run.setdefault("name", agent_name or "")
    if model_override:
        agent_for_run["model"] = model_override
    if reasoning_override in {"low", "medium", "high"}:
        agent_for_run["reasoning_effort"] = reasoning_override

    try:
        result = run_agent_via_responses(agent_for_run, request_payload)
    except AgentRunnerError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})
    except Exception as exc:  # pragma: no cover - bubble unexpected errors
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    body = {
        "ok": True,
        "result": result,
        "auto_correct": bool(request_payload.get("auto_correct")),
    }
    return JSONResponse(body)
