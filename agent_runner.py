#!/usr/bin/env python3
"""Helpers for executing stored agents via the OpenAI Responses API."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:
    from openai import OpenAI, BadRequestError
except ImportError as exc:  # pragma: no cover
    OpenAI = None  # type: ignore
    BadRequestError = None  # type: ignore
    _import_error = exc  # type: ignore
else:
    _import_error = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from openai.types.responses import Response  # type: ignore
else:  # pragma: no cover
    Response = Any  # type: ignore


load_dotenv()

DEFAULT_MODEL = (
    os.getenv("OPENAI_DEFAULT_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "gpt-4.1-mini"
)
DEFAULT_LANGUAGE_HINT = os.getenv("OPENAI_LANGUAGE_HINT") or "cs"

REASONING_EFFORT_VALUES = {"low", "medium", "high"}
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _get_model_capabilities(model: str) -> Dict[str, bool]:
    normalized = (model or "").strip().lower()
    for prefix in REASONING_PREFIXES:
        if not prefix:
            continue
        lowered = prefix.lower()
        if normalized == lowered or normalized.startswith(f"{lowered}-"):
            return {"temperature": False, "top_p": False, "reasoning": True}
    return {"temperature": True, "top_p": True, "reasoning": False}


def _normalize_reasoning_effort(value: Optional[Any]) -> str:
    if value is None:
        return "medium"
    normalized = str(value).strip().lower()
    return normalized if normalized in REASONING_EFFORT_VALUES else "medium"

BLOCK_TAGS = ("h1", "h2", "h3", "p", "div", "small", "note", "blockquote", "li")


class AgentRunnerError(RuntimeError):
    """Raised when an agent cannot be executed due to configuration issues."""


_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return a cached OpenAI client, validating prerequisites first."""
    global _client
    if _client is not None:
        return _client
    if OpenAI is None:
        reason = f"ImportError: {_import_error}" if _import_error else ""
        message = "Knihovna 'openai' není nainstalovaná. Přidejte ji do prostředí."
        if reason:
            message = f"{message} ({reason})"
        raise AgentRunnerError(message)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AgentRunnerError(
            "Chybí proměnná prostředí OPENAI_API_KEY. Uložte ji do .env nebo prostředí."
        )
    _client = OpenAI(api_key=api_key)
    return _client


def _extract_output_text(response: Response) -> str:
    """Safely extract concatenated text output from a Responses API reply."""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    data: Dict[str, Any]
    try:
        data = response.model_dump()
    except AttributeError:
        data = getattr(response, "__dict__", {})

    output_chunks = []
    output = data.get("output") or data.get("outputs") or []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or []
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type not in {"text", "output_text"}:
                    continue
                text_payload = block.get("text")
                if isinstance(text_payload, dict):
                    value = text_payload.get("value") or text_payload.get("text")
                    if value:
                        output_chunks.append(str(value))
                elif isinstance(text_payload, str):
                    output_chunks.append(text_payload)
    return "\n".join(output_chunks).strip()


def _extract_stop_reason(response_dict: Dict[str, Any]) -> Optional[str]:
    output = response_dict.get("output") or response_dict.get("outputs") or []
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                reason = item.get("stop_reason")
                if reason:
                    return str(reason)
    return None


def _extract_usage(response_dict: Dict[str, Any]) -> Dict[str, Any]:
    usage = response_dict.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    allowed = ("input_tokens", "output_tokens", "total_tokens")
    return {key: usage[key] for key in allowed if key in usage}


def _infer_block_type(element) -> str:
    name = (element.name or "").lower()
    if name in {"h1", "h2", "h3"}:
        return name
    if name == "p":
        has_small_child = False
        for child in element.contents:
            if getattr(child, "name", None):
                child_name = child.name.lower()
                if child_name == "small":
                    has_small_child = True
                    continue
                if child_name == "br":
                    continue
                return "p"
            else:
                if isinstance(child, str) and child.strip():
                    return "p"
        if has_small_child:
            return "small"
        return "p"
    if name == "small":
        return "small"
    if name == "note":
        return "note"
    if name == "blockquote":
        return "blockquote"
    if name == "li":
        return "li"
    if name == "div":
        classes = element.get("class") or []
        normalized = [cls.lower() for cls in classes]
        if "centered" in normalized:
            return "centered"
        if "note" in normalized:
            return "note"
        return "p"
    return "p"


def _html_to_blocks(html_text: str) -> list[Dict[str, str]]:
    soup = BeautifulSoup(html_text or "", "html.parser")
    if soup is None:
        return []
    blocks: list[Dict[str, str]] = []
    seen = set()

    for element in soup.find_all(BLOCK_TAGS):
        parent = element.find_parent(BLOCK_TAGS)
        if parent:
            continue
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        block_type = _infer_block_type(element)
        token = id(element)
        if token in seen:
            continue
        seen.add(token)
        blocks.append({
            "type": block_type,
            "text": text,
        })

    if not blocks:
        fallback_text = soup.get_text(" ", strip=True)
        if fallback_text:
            blocks.append({"type": "p", "text": fallback_text})

    return [
        {"id": f"b{index}", "type": block["type"], "text": block["text"]}
        for index, block in enumerate(blocks, start=1)
    ]


def _build_document_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    python_html = payload.get("python_html")
    if not python_html or not str(python_html).strip():
        raise AgentRunnerError("Python výstup pro agenta je prázdný.")

    blocks = _html_to_blocks(str(python_html))
    if not blocks:
        raise AgentRunnerError("Python výstup se nepodařilo převést na bloky.")

    language_hint = payload.get("language_hint") or DEFAULT_LANGUAGE_HINT

    page_meta: Dict[str, Any] = {}
    provided_meta = payload.get("page_meta")
    if isinstance(provided_meta, dict):
        page_meta.update({
            key: value
            for key, value in provided_meta.items()
            if value not in (None, "", [], {})
        })

    for key in ("page_uuid", "book_uuid", "book_title", "page_number", "page_index"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            page_meta.setdefault(key, value)

    if "page" not in page_meta:
        candidate = payload.get("page_number") or page_meta.get("page_number")
        if candidate not in (None, "", [], {}):
            page_meta["page"] = candidate
        elif isinstance(page_meta.get("page_index"), (int, float)):
            page_meta["page"] = page_meta["page_index"]

    if payload.get("book_title") and "work" not in page_meta:
        page_meta["work"] = payload["book_title"]

    return {
        "language_hint": language_hint,
        "page_meta": page_meta,
        "blocks": blocks,
    }


def run_agent(agent: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a stored agent configuration via the Responses API."""
    if not isinstance(agent, dict):
        raise AgentRunnerError("Agent nemá očekávaný formát.")

    prompt = str(agent.get("prompt") or "").strip()
    if not prompt:
        raise AgentRunnerError("Agent nemá vyplněný prompt.")

    document_payload = _build_document_payload(payload or {})
    user_payload = json.dumps(document_payload, ensure_ascii=False, indent=2)

    model = (
        str(agent.get("model")).strip()
        if agent.get("model")
        else DEFAULT_MODEL
    )

    client = _get_client()

    capabilities = _get_model_capabilities(model)
    temperature = agent.get("temperature")
    top_p = agent.get("top_p")
    reasoning_effort_value = payload.get("reasoning_effort") or agent.get("reasoning_effort")
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort_value)
    supports_temperature = capabilities.get("temperature", True)
    supports_top_p = capabilities.get("top_p", True)
    supports_reasoning = capabilities.get("reasoning", False)

    request_kwargs: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_payload}],
            },
        ],
    }
    if supports_temperature and temperature is not None:
        try:
            request_kwargs["temperature"] = float(temperature)
        except (TypeError, ValueError):
            pass
    elif not supports_temperature and temperature is not None:
        print(f"[AgentDebug] Model {model} ignoruje parametr temperature – nebude odeslán.")
    if supports_top_p and top_p is not None:
        try:
            request_kwargs["top_p"] = float(top_p)
        except (TypeError, ValueError):
            pass
    elif not supports_top_p and top_p is not None:
        print(f"[AgentDebug] Model {model} ignoruje parametr top_p – nebude odeslán.")
    if supports_reasoning:
        request_kwargs["reasoning"] = {"effort": normalized_reasoning_effort}
        agent["reasoning_effort"] = normalized_reasoning_effort

    debug_agent = agent.get("name") or agent.get("display_name") or "unknown"
    print("\n=== [AgentDebug] OpenAI Request ===")
    print(f"Agent: {debug_agent}")
    print(f"Model: {request_kwargs.get('model')}")
    if "temperature" in request_kwargs:
        print(f"Temperature: {request_kwargs.get('temperature')}")
    if "top_p" in request_kwargs:
        print(f"Top P: {request_kwargs.get('top_p')}")
    if "reasoning" in request_kwargs:
        print(f"Reasoning effort: {request_kwargs['reasoning'].get('effort')}")
    print("--- Prompt ---")
    for part in request_kwargs.get("input", []):
        role = part.get("role")
        contents = part.get("content") or []
        for block in contents:
            if block.get("type") == "input_text":
                print(f"[{role}] {block.get('text')}")
    print("=== [AgentDebug] End Request ===\n")

    def _retry_without_unsupported(error: Exception) -> Optional[Response]:
        if BadRequestError is None or not isinstance(error, BadRequestError):
            return None
        message = " ".join(str(part) for part in error.args if part)
        message_lower = message.lower()
        removed_any = False
        if "temperature" in request_kwargs and "temperature" in message_lower:
            print("[AgentDebug] Model nepodporuje parametr temperature – opakuji bez něj.")
            request_kwargs.pop("temperature", None)
            removed_any = True
        if "top_p" in request_kwargs and "top_p" in message_lower:
            print("[AgentDebug] Model nepodporuje parametr top_p – opakuji bez něj.")
            request_kwargs.pop("top_p", None)
            removed_any = True
        if "reasoning" in request_kwargs and "reasoning" in message_lower:
            print("[AgentDebug] Model nepodporuje pole reasoning – opakuji bez něj.")
            request_kwargs.pop("reasoning", None)
            removed_any = True
        if not removed_any:
            return None
        return client.responses.create(**request_kwargs)

    try:
        response = client.responses.create(**request_kwargs)
    except Exception as exc:
        retry = _retry_without_unsupported(exc)
        if retry is None:
            raise
        response = retry

    try:
        response_dict = response.model_dump()
    except AttributeError:
        response_dict = getattr(response, "__dict__", {})

    print("=== [AgentDebug] OpenAI Response ===")
    try:
        print(response.output_text)
    except Exception:
        try:
            print(json.dumps(response_dict, ensure_ascii=False, indent=2))
        except Exception:
            print("[AgentDebug] Nelze zobrazit odpověď.")
    print("=== [AgentDebug] End Response ===\n")

    text = _extract_output_text(response)
    stop_reason = _extract_stop_reason(response_dict)
    usage = _extract_usage(response_dict)

    return {
        "text": text,
        "response_id": response_dict.get("id") or getattr(response, "id", None),
        "model": response_dict.get("model") or model,
        "stop_reason": stop_reason,
        "usage": usage,
        "input_document": document_payload,
        "reasoning_effort": agent.get("reasoning_effort"),
    }
