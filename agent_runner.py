#!/usr/bin/env python3
"""Helpers for executing stored agents via the OpenAI Responses API."""

from __future__ import annotations

import copy
import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

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


def _clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < minimum:
        return minimum
    if number > maximum:
        return maximum
    return number


def _extract_settings(agent: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    settings = agent.get("settings")
    defaults: Dict[str, Any] = {}
    per_model: Dict[str, Dict[str, Any]] = {}
    if isinstance(settings, dict):
        raw_defaults = settings.get("defaults")
        if isinstance(raw_defaults, dict):
            defaults = raw_defaults
        raw_per_model = settings.get("per_model")
        if isinstance(raw_per_model, dict):
            per_model = raw_per_model  # type: ignore[assignment]
    return defaults, per_model


def _get_effective_settings(agent: Dict[str, Any], model: str) -> Dict[str, Any]:
    defaults, per_model = _extract_settings(agent)
    model_settings = per_model.get(model) if isinstance(per_model.get(model), dict) else {}
    capabilities = _get_model_capabilities(model)
    result: Dict[str, Any] = {}
    if capabilities.get("temperature"):
        if isinstance(model_settings, dict) and "temperature" in model_settings:
            result["temperature"] = _clamp_float(model_settings["temperature"], 0.0, 2.0, 0.0)
        elif "temperature" in defaults:
            result["temperature"] = _clamp_float(defaults["temperature"], 0.0, 2.0, 0.0)
        elif "temperature" in agent:
            result["temperature"] = _clamp_float(agent["temperature"], 0.0, 2.0, 0.0)
    if capabilities.get("top_p"):
        if isinstance(model_settings, dict) and "top_p" in model_settings:
            result["top_p"] = _clamp_float(model_settings["top_p"], 0.0, 1.0, 1.0)
        elif "top_p" in defaults:
            result["top_p"] = _clamp_float(defaults["top_p"], 0.0, 1.0, 1.0)
        elif "top_p" in agent:
            result["top_p"] = _clamp_float(agent["top_p"], 0.0, 1.0, 1.0)
    if capabilities.get("reasoning"):
        reasoning_source = None
        if isinstance(model_settings, dict) and "reasoning_effort" in model_settings:
            reasoning_source = model_settings["reasoning_effort"]
        elif "reasoning_effort" in defaults:
            reasoning_source = defaults["reasoning_effort"]
        elif "reasoning_effort" in agent:
            reasoning_source = agent["reasoning_effort"]
        result["reasoning_effort"] = _normalize_reasoning_effort(reasoning_source)
    return result

BLOCK_TAGS = ("h1", "h2", "h3", "p", "div", "small", "note", "blockquote", "li")


class AgentRunnerError(RuntimeError):
    """Raised when an agent cannot be executed due to configuration issues."""


class AgentDiffApplicationError(RuntimeError):
    """Raised when diff-based agent output cannot be applied."""


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


def _strip_code_fences(text: str) -> str:
    """Remove common Markdown code fences to simplify JSON parsing."""
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped.split("\n", 1)
        if len(inner) == 2:
            payload = inner[1]
            closing_index = payload.rfind("\n```")
            if closing_index != -1:
                return payload[:closing_index].strip()
        return stripped.strip("`").strip()
    return stripped


def _safe_json_loads(candidate: str) -> Optional[Dict[str, Any]]:
    """Parse JSON if possible, returning None on failure."""
    if not candidate or not isinstance(candidate, str):
        return None
    text = _strip_code_fences(candidate)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _log_diff_warning(payload: Dict[str, Any], message: str) -> None:
    """Print a warning about diff processing failure including page context."""
    if not isinstance(payload, dict):
        payload = {}
    page_meta = {}
    if isinstance(payload.get("page_meta"), dict):
        page_meta = payload["page_meta"]
    details = []
    page_uuid = (
        payload.get("page_uuid")
        or payload.get("page_id")
        or page_meta.get("page_uuid")
        or page_meta.get("uuid")
    )
    if page_uuid:
        details.append(f"page_uuid={page_uuid}")
    page_label = (
        payload.get("page_number")
        or payload.get("page_index")
        or page_meta.get("page")
        or page_meta.get("page_number")
    )
    if page_label not in (None, "", []):
        details.append(f"page_ref={page_label}")
    context = ", ".join(details) if details else "page_context=unknown"
    print(f"[AgentDiff] {message} ({context})")


def _apply_diff_to_document(
    document: Dict[str, Any],
    diff_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a diff `{changes: [...]}` to the original document payload."""
    if not isinstance(document, dict):
        raise AgentDiffApplicationError("Vstupní dokument má neočekávaný formát.")
    original_blocks = document.get("blocks")
    if not isinstance(original_blocks, list):
        raise AgentDiffApplicationError("Vstupní dokument neobsahuje pole 'blocks'.")
    changes = diff_payload.get("changes")
    if not isinstance(changes, list):
        raise AgentDiffApplicationError("Diff výstup postrádá pole 'changes'.")

    # Deep copy input blocks so we don't mutate cached originals.
    cloned_blocks = []
    block_index_map: Dict[str, Dict[str, Any]] = {}
    for block in original_blocks:
        cloned = copy.deepcopy(block)
        cloned_blocks.append(cloned)
        block_id = cloned.get("id")
        if isinstance(block_id, str):
            block_index_map[block_id] = cloned

    removed_ids: set[str] = set()

    for change in changes:
        if not isinstance(change, dict):
            raise AgentDiffApplicationError("Položka v 'changes' není objekt.")
        block_id = change.get("id")
        if not isinstance(block_id, str) or not block_id.strip():
            raise AgentDiffApplicationError("Položka diffu postrádá platné 'id'.")
        target = block_index_map.get(block_id)
        if target is None:
            raise AgentDiffApplicationError(f"Diff odkazuje na neznámý blok '{block_id}'.")
        block_type = str(target.get("type") or "").lower()
        if block_type == "note":
            # Notes slouží jen jako kontext – ignorujeme změny i smazání.
            continue
        if "text" not in change:
            # Bez textu není co aplikovat; přeskočíme tichou úpravou.
            continue
        raw_text = change.get("text")
        text_value = "" if raw_text is None else str(raw_text)
        if not text_value.strip():
            removed_ids.add(block_id)
            continue
        target["text"] = text_value

    if not removed_ids and all(block.get("text") == original.get("text") for block, original in zip(cloned_blocks, original_blocks)):
        # No effective changes – return original clone to keep structure uniform.
        result_document = copy.deepcopy(document)
        result_document["blocks"] = cloned_blocks
        return result_document

    result_blocks = [
        block for block in cloned_blocks
        if not isinstance(block.get("id"), str) or block.get("id") not in removed_ids
    ]

    result_document = copy.deepcopy(document)
    result_document["blocks"] = result_blocks
    return result_document


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
    collection = str(payload.get("collection") or "").strip().lower()
    if collection == "joiners":
        context = payload.get("stitch_context")
        if not context:
            raise AgentRunnerError("Chybí data pro napojení stran.")
        try:
            serialized = json.loads(json.dumps(context, ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise AgentRunnerError(f"Kontext pro napojení stran není validní JSON: {exc}") from exc
        if isinstance(serialized, dict) and "language_hint" not in serialized:
            serialized["language_hint"] = payload.get("language_hint") or DEFAULT_LANGUAGE_HINT
        return serialized

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
    effective_settings = _get_effective_settings(agent, model)
    payload_temperature = payload.get("temperature") if isinstance(payload, dict) else None
    payload_top_p = payload.get("top_p") if isinstance(payload, dict) else None
    payload_reasoning = payload.get("reasoning_effort") if isinstance(payload, dict) else None

    temperature = None
    if capabilities.get("temperature"):
        if payload_temperature is not None:
            temperature = _clamp_float(payload_temperature, 0.0, 2.0, effective_settings.get("temperature", 0.0))
        else:
            temperature = effective_settings.get("temperature")

    top_p = None
    if capabilities.get("top_p"):
        if payload_top_p is not None:
            top_p = _clamp_float(payload_top_p, 0.0, 1.0, effective_settings.get("top_p", 1.0))
        else:
            top_p = effective_settings.get("top_p")

    reasoning_source = payload_reasoning or effective_settings.get("reasoning_effort")
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_source)
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
        request_kwargs["temperature"] = _clamp_float(
            temperature,
            0.0,
            2.0,
            effective_settings.get("temperature", 0.0),
        )
    elif not supports_temperature and payload_temperature is not None:
        print(f"[AgentDebug] Model {model} ignoruje parametr temperature – nebude odeslán.")
    if supports_top_p and top_p is not None:
        request_kwargs["top_p"] = _clamp_float(
            top_p,
            0.0,
            1.0,
            effective_settings.get("top_p", 1.0),
        )
    elif not supports_top_p and payload_top_p is not None:
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
    diff_applied = False
    diff_changes = None
    output_document: Optional[Dict[str, Any]] = None

    parsed_output = _safe_json_loads(text)
    blocks_present = isinstance(document_payload.get("blocks"), list)
    if blocks_present and parsed_output:
        if isinstance(parsed_output.get("changes"), list):
            try:
                applied_document = _apply_diff_to_document(document_payload, parsed_output)
            except AgentDiffApplicationError as exc:
                _log_diff_warning(document_payload, f"Diff aplikace selhala: {exc}")
                text = json.dumps(document_payload, ensure_ascii=False, indent=2)
            else:
                text = json.dumps(applied_document, ensure_ascii=False, indent=2)
                diff_applied = True
                diff_changes = copy.deepcopy(parsed_output.get("changes"))
                output_document = applied_document
        elif isinstance(parsed_output.get("blocks"), list):
            # Agent vrátil plný dokument – znormalizujeme formátování.
            text = json.dumps(parsed_output, ensure_ascii=False, indent=2)
            output_document = parsed_output
        else:
            text = text.strip()
    elif parsed_output and isinstance(parsed_output.get("blocks"), list):
        # I když nemáme 'blocks' v document_payload (např. joiner snapshot),
        # uchováme výsledek pro případné další zpracování na klientovi.
        output_document = parsed_output
        text = json.dumps(parsed_output, ensure_ascii=False, indent=2)

    return {
        "text": text,
        "response_id": response_dict.get("id") or getattr(response, "id", None),
        "model": response_dict.get("model") or model,
        "stop_reason": stop_reason,
        "usage": usage,
        "input_document": document_payload,
        "reasoning_effort": agent.get("reasoning_effort"),
        "diff_applied": diff_applied,
        "diff_changes": diff_changes if diff_applied else None,
        "output_document": output_document,
    }
