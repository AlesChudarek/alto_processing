#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webový server pro porovnání původního TypeScript a nového Python zpracování ALTO
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import json
import requests
import subprocess
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import urlparse, parse_qs
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import html
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Import původního procesoru
from main_processor import AltoProcessor, DEFAULT_API_BASES
from agent_runner import (
    AgentRunnerError,
    run_agent as run_agent_via_responses,
    DEFAULT_MODEL,
    REASONING_PREFIXES,
)


ROOT_DIR = Path(__file__).resolve().parent
TS_DIST_PATH = ROOT_DIR / 'dist' / 'run_original.js'
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 1200
DEFAULT_AGENT_PROMPT_TEXT = (
    "Jsi pečlivý korektor češtiny. Dostaneš JSON s klíči "
    "\"language_hint\", \"page_meta\" a \"blocks\". Blocks je pole "
    "objektů {id, type, text}. Oprav překlepy a zjevné OCR chyby pouze "
    "v hodnotách \"text\". Nesjednocuj styl, neměň typy bloků ani jejich "
    "pořadí. Zachovej diakritiku, pokud lze. Odpovídej pouze validním "
    "JSON se stejnou strukturou a klíči jako vstup."
)

# Style value reused for <note> blocks in agent diff rendering.
NOTE_STYLE_ATTR = 'display:block;font-size:0.82em;color:#1e5aa8;font-weight:bold;'

KNOWN_LIBRARY_OVERRIDES: Dict[str, Dict[str, str]] = {
    "https://kramerius.mzk.cz/search/api/v5.0": {
        "code": "mzk",
        "label": "Moravská zemská knihovna v Brně",
        "handle_base": "https://kramerius.mzk.cz/search",
    },
    "https://kramerius5.nkp.cz/search/api/v5.0": {
        "code": "nkp",
        "label": "Národní knihovna České republiky",
        "handle_base": "https://kramerius5.nkp.cz/search",
    },
}

DIFF_MODE_WORD = 'word'
DIFF_MODE_CHAR = 'char'
DIFF_MODE_NONE = 'none'

_VOID_HTML_ELEMENTS = {
    'br', 'hr', 'img', 'input', 'meta', 'link', 'source', 'track',
    'area', 'col', 'embed', 'param', 'wbr', 'base'
}
_TAG_REGEX = re.compile(r'<[^>]+?>')
_TAG_NAME_REGEX = re.compile(r'^<\s*/?\s*([a-zA-Z0-9:-]+)')
_WORD_SPLIT_REGEX = re.compile(r'(\s+|[^\w]+)', re.UNICODE)


@dataclass(frozen=True)
class HtmlToken:
    kind: str  # 'tag' or 'text'
    value: str
    is_whitespace: bool = False


def _normalize_diff_mode(mode: Optional[str]) -> str:
    if mode == DIFF_MODE_CHAR:
        return DIFF_MODE_CHAR
    return DIFF_MODE_WORD


def _extract_tag_name(tag_text: str) -> str:
    match = _TAG_NAME_REGEX.match(tag_text)
    return match.group(1).lower() if match else ''


def split_html_blocks(html_content: str) -> List[str]:
    if not html_content:
        return []

    blocks: List[str] = []
    depth = 0
    block_start: Optional[int] = None
    last_index = 0

    for match in _TAG_REGEX.finditer(html_content):
        start, end = match.span()
        tag_text = match.group(0)

        if depth == 0 and start > last_index:
            segment = html_content[last_index:start]
            if segment:
                blocks.append(segment)

        tag_name = _extract_tag_name(tag_text)
        is_closing = tag_text.startswith('</')
        is_self_closing = tag_text.endswith('/>') or tag_name in _VOID_HTML_ELEMENTS

        if not is_closing and not is_self_closing:
            if depth == 0:
                block_start = start
            depth += 1
        elif is_self_closing:
            if depth == 0:
                blocks.append(tag_text)
        else:
            depth = max(depth - 1, 0)
            if depth == 0 and block_start is not None:
                blocks.append(html_content[block_start:end])
                block_start = None

        last_index = end

    if block_start is not None and block_start < len(html_content):
        blocks.append(html_content[block_start:])
    elif last_index < len(html_content):
        remainder = html_content[last_index:]
        if remainder:
            blocks.append(remainder)

    if not blocks and html_content:
        return [html_content]
    return blocks


def split_text_tokens(text: str, mode: str) -> List[HtmlToken]:
    if not text:
        return []

    if mode == DIFF_MODE_CHAR:
        return [
            HtmlToken('text', ch, ch.isspace())
            for ch in text
        ]

    tokens: List[HtmlToken] = []
    last_index = 0
    for match in _WORD_SPLIT_REGEX.finditer(text):
        start, end = match.span()
        if start > last_index:
            chunk = text[last_index:start]
            tokens.append(HtmlToken('text', chunk, chunk.isspace()))
        chunk = match.group(0)
        if chunk:
            tokens.append(HtmlToken('text', chunk, chunk.isspace()))
        last_index = end
    if last_index < len(text):
        chunk = text[last_index:]
        tokens.append(HtmlToken('text', chunk, chunk.isspace()))
    return tokens


def tokenize_block(block_html: str, mode: str) -> List[HtmlToken]:
    if not block_html:
        return []

    tokens: List[HtmlToken] = []
    last_index = 0
    for match in _TAG_REGEX.finditer(block_html):
        start, end = match.span()
        if start > last_index:
            tokens.extend(split_text_tokens(block_html[last_index:start], mode))
        tokens.append(HtmlToken('tag', match.group(0), False))
        last_index = end
    if last_index < len(block_html):
        tokens.extend(split_text_tokens(block_html[last_index:], mode))
    return tokens


def render_tokens(tokens: List[HtmlToken], highlight: Optional[str] = None) -> str:
    parts: List[str] = []
    for token in tokens:
        if token.kind == 'tag':
            parts.append(token.value)
            continue
        if highlight in ('added', 'removed') and not token.is_whitespace and token.value:
            parts.append(f'<span class="diff-{highlight}">{token.value}</span>')
        else:
            parts.append(token.value)
    return ''.join(parts)


def diff_block_content(left_block: str, right_block: str, mode: str) -> Tuple[str, str]:
    mode_normalized = _normalize_diff_mode(mode)
    left_tokens = tokenize_block(left_block, mode_normalized)
    right_tokens = tokenize_block(right_block, mode_normalized)
    matcher = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)

    left_parts: List[str] = []
    right_parts: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left_slice = left_tokens[i1:i2]
        right_slice = right_tokens[j1:j2]

        if tag == 'equal':
            left_parts.append(render_tokens(left_slice))
            right_parts.append(render_tokens(right_slice))
        elif tag == 'delete':
            left_parts.append(render_tokens(left_slice, 'removed'))
        elif tag == 'insert':
            right_parts.append(render_tokens(right_slice, 'added'))
        elif tag == 'replace':
            left_parts.append(render_tokens(left_slice, 'removed'))
            right_parts.append(render_tokens(right_slice, 'added'))

    return ''.join(left_parts), ''.join(right_parts)


def wrap_block(content: str, change: str) -> str:
    classes = ['diff-block']
    if change == 'added':
        classes.append('diff-block--added')
    elif change == 'removed':
        classes.append('diff-block--removed')
    elif change == 'changed':
        classes.append('diff-block--changed')
    class_attr = ' '.join(classes)
    data_attr = f' data-diff-change="{change}"' if change != 'equal' else ''
    return f'<div class="{class_attr}"{data_attr}>{content}</div>'


def wrap_diff_content(blocks: List[str], mode: str) -> str:
    mode_attr = f' data-diff-mode="{mode}"' if mode in {DIFF_MODE_WORD, DIFF_MODE_CHAR} else ''
    inner = ''.join(blocks)
    if not inner:
        inner = '<div class="diff-placeholder">Bez obsahu</div>'
    return f'<div class="diff-content diff-html"{mode_attr}>{inner}</div>'


def _annotate_block_classes(block_html: str, change: str) -> str:
    if not block_html or not change or change == 'equal':
        return block_html
    class_map = {
        'added': 'diff-block diff-block--added',
        'removed': 'diff-block diff-block--removed',
        'changed': 'diff-block diff-block--changed',
    }
    class_name = class_map.get(change)
    if not class_name:
        return block_html
    stripped = block_html.lstrip()
    prefix_len = len(block_html) - len(stripped)
    prefix = block_html[:prefix_len]
    target = stripped
    leading_inline = ''
    while target.lower().startswith('<br'):
        end_idx = target.find('>')
        if end_idx == -1:
            break
        leading_inline += target[:end_idx + 1]
        target = target[end_idx + 1:]
    if not target:
        return prefix + leading_inline
    match = re.match(r'<([a-zA-Z0-9:-]+)([^>]*)>', target)
    if match:
        tag_name = match.group(1)
        attrs = match.group(2) or ''
        if 'class=' in attrs:
            updated = re.sub(r'class="([^"]*)"', lambda m: f'class="{m.group(1)} {class_name}"', target, count=1)
            return prefix + leading_inline + updated
        else:
            separator = ''
            if not attrs or not attrs.endswith(' '):
                separator = ' '
            insertion = f'<{tag_name}{attrs}{separator}class="{class_name}">'
            updated = target.replace(match.group(0), insertion, 1)
            return prefix + leading_inline + updated
    remaining = block_html[prefix_len + len(leading_inline):]


def build_html_diff(python_html: str, ts_html: str, mode: str) -> Dict[str, str]:
    mode_normalized = _normalize_diff_mode(mode)
    python_blocks = split_html_blocks(python_html or '')
    ts_blocks = split_html_blocks(ts_html or '')

    matcher = SequenceMatcher(None, python_blocks, ts_blocks, autojunk=False)

    python_render: List[str] = []
    ts_render: List[str] = []
    changes_detected = False

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        py_slice = python_blocks[i1:i2]
        ts_slice = ts_blocks[j1:j2]

        if tag == 'equal':
            for block in py_slice:
                python_render.append(wrap_block(block, 'equal'))
            for block in ts_slice:
                ts_render.append(wrap_block(block, 'equal'))
            continue

        changes_detected = True

        if tag == 'delete':
            for block in py_slice:
                python_render.append(wrap_block(block, 'added'))
            continue
        if tag == 'insert':
            for block in ts_slice:
                ts_render.append(wrap_block(block, 'removed'))
            continue

        if len(py_slice) == 1 and len(ts_slice) == 1:
            left_markup, right_markup = diff_block_content(py_slice[0], ts_slice[0], mode_normalized)
            python_render.append(wrap_block(left_markup, 'changed'))
            ts_render.append(wrap_block(right_markup, 'changed'))
        else:
            for block in py_slice:
                python_render.append(wrap_block(block, 'added'))
            for block in ts_slice:
                ts_render.append(wrap_block(block, 'removed'))

    return {
        'python': wrap_diff_content(python_render, mode_normalized),
        'typescript': wrap_diff_content(ts_render, mode_normalized),
        'has_changes': changes_detected,
        'mode': mode_normalized,
    }


def block_to_html_from_dict(block: dict) -> Optional[str]:
    if not isinstance(block, dict):
        return None
    text = block.get('text')
    if text is None:
        return None
    text_str = str(text).strip()
    if not text_str:
        return None

    block_type = str(block.get('type') or '').lower()
    escaped_text = html.escape(text_str, quote=False)

    tag = 'p'
    attrs = {}
    classes: List[str] = []
    style_attr = None
    wrap_with_small = False
    prefix = ''

    block_id = block.get('id')
    if block_id not in (None, ''):
        attrs['data-block-id'] = str(block_id)

    if block_type in {'h1', 'h2', 'h3'}:
        tag = block_type
    elif block_type == 'small':
        wrap_with_small = True
    elif block_type == 'note':
        tag = 'note'
        style_attr = NOTE_STYLE_ATTR
        prefix = '<br>'
    elif block_type == 'centered':
        tag = 'div'
        classes.append('centered')
    elif block_type == 'blockquote':
        tag = 'blockquote'
    elif block_type == 'li':
        tag = 'p'
        attrs['data-block-type'] = 'li'
    else:
        tag = 'p'
        if block_type and block_type != 'p':
            attrs['data-block-type'] = block_type

    if classes:
        attrs['class'] = ' '.join(classes)
    if style_attr:
        attrs['style'] = style_attr

    attr_parts = []
    for key, value in attrs.items():
        attr_parts.append(f'{key}="{html.escape(str(value), quote=False)}"')
    attr_str = f" {' '.join(attr_parts)}" if attr_parts else ''

    if wrap_with_small:
        markup = f'<p{attr_str}><small>{escaped_text}</small></p>'
    else:
        markup = f'<{tag}{attr_str}>{escaped_text}</{tag}>'
        if tag == 'note':
            markup = prefix + markup
    return markup


def build_agent_diff(original_html: str, corrected_html: str, mode: str) -> Dict[str, str]:
    mode_normalized = _normalize_diff_mode(mode)

    def parse_blocks(html_text: str) -> List[str]:
        try:
            data = json.loads(html_text)
            if isinstance(data, dict) and isinstance(data.get('blocks'), list):
                output = []
                for block in data['blocks']:
                    markup = block_to_html_from_dict(block)
                    if markup:
                        output.append((block.get('id'), markup))
                if output:
                    return output
        except Exception:
            pass
        tokens = split_html_blocks(html_text or '')
        fallback: List[Tuple[Optional[str], str]] = []
        for token in tokens:
            if not token:
                continue
            block_id = None
            match = re.search(r'data-block-id="([^"]+)"', token)
            if match:
                block_id = match.group(1)
            fallback.append((block_id, token))
        return fallback

    original_pairs = parse_blocks(original_html or '')
    corrected_pairs = parse_blocks(corrected_html or '')

    original_map = {bid: markup for bid, markup in original_pairs if bid}
    corrected_map = {bid: markup for bid, markup in corrected_pairs if bid}

    processed_ids = set()
    original_render: List[str] = []
    corrected_render: List[str] = []
    changes_detected = False

    for bid, markup in original_pairs:
        if bid and bid in corrected_map:
            corrected_markup = corrected_map[bid]
            if markup == corrected_markup:
                original_render.append(markup)
                corrected_render.append(corrected_markup)
            else:
                changes_detected = True
                left_markup, right_markup = diff_block_content(markup, corrected_markup, mode_normalized)
                original_render.append(_annotate_block_classes(left_markup, 'changed'))
                corrected_render.append(_annotate_block_classes(right_markup, 'changed'))
            processed_ids.add(bid)
        else:
            changes_detected = True
            original_render.append(_annotate_block_classes(markup, 'removed'))

    for bid, markup in corrected_pairs:
        if bid and bid in processed_ids:
            continue
        if bid and bid in original_map:
            continue
        changes_detected = True
        corrected_render.append(_annotate_block_classes(markup, 'added'))

    return {
        'original': ''.join(original_render),
        'corrected': ''.join(corrected_render),
        'has_changes': changes_detected,
        'mode': mode_normalized,
    }


# Agents storage/helpers
AGENTS_DIR = ROOT_DIR / 'agents'
AGENT_NAME_RE = re.compile(r'^[A-Za-z0-9._-]{1,64}$')

def ensure_agents_dir():
    try:
        AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def safe_agent_name(name: str) -> Optional[str]:
    if not name or not isinstance(name, str):
        return None
    nm = name.strip()
    if AGENT_NAME_RE.match(nm):
        return nm
    return None


def sanitize_agent_name(name: str) -> Optional[str]:
    """Produce a filesystem-safe agent filename from arbitrary display name.
    Returns None if result would be empty."""
    if not name or not isinstance(name, str):
        return None
    # replace spaces and disallowed chars with -
    s = name.strip()
    # normalize: replace long sequences of non-allowed chars with '-'
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', s)
    s = re.sub(r'-{2,}', '-', s)
    s = s.strip('-')
    if not s:
        return None
    # limit length
    if len(s) > 64:
        s = s[:64]
    # ensure it matches the allowed pattern
    if AGENT_NAME_RE.match(s):
        return s
    return None

def _is_reasoning_model_id(model: str) -> bool:
    normalized = (model or "").strip().lower()
    if not normalized:
        return False
    for prefix in REASONING_PREFIXES:
        lowered = prefix.lower()
        if normalized == lowered or normalized.startswith(f"{lowered}-"):
            return True
    return False

def _clamp_float(value, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < minimum:
        return minimum
    if number > maximum:
        return maximum
    return number

def _normalize_reasoning_effort_value(value, default: str = "medium") -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    return normalized if normalized in {"low", "medium", "high"} else default

def _supports_temperature(model: str) -> bool:
    return not _is_reasoning_model_id(model)

def _supports_top_p(model: str) -> bool:
    return not _is_reasoning_model_id(model)

def _supports_reasoning(model: str) -> bool:
    return _is_reasoning_model_id(model)


def agent_filepath(name: str) -> Path:
    return AGENTS_DIR / f"{name}.json"

def list_agents_files():
    ensure_agents_dir()
    out = []
    for p in sorted(AGENTS_DIR.glob('*.json')):
        try:
            stat = p.stat()
            parsed = None
            display_name = p.stem
            try:
                raw = p.read_text(encoding='utf-8')
                parsed = json.loads(raw)
                display_name = parsed.get('display_name') or parsed.get('name') or p.stem
            except Exception:
                parsed = None
                display_name = p.stem
            out.append({
                'name': p.stem,
                'display_name': display_name,
                'agent': parsed,
                'path': str(p),
                'updated_at': stat.st_mtime,
                'size': stat.st_size,
            })
        except Exception:
            continue
    return out

def read_agent_file(name: str) -> Optional[dict]:
    nm = safe_agent_name(name)
    if not nm:
        return None
    p = agent_filepath(nm)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

def write_agent_file(data: dict) -> bool:
    name = data.get('name') if isinstance(data, dict) else None
    nm = safe_agent_name(name)
    display_name = name if isinstance(name, str) else ''
    if not nm:
        nm = sanitize_agent_name(display_name)
    if not nm:
        return False
    ensure_agents_dir()
    p = agent_filepath(nm)
    # sanitize and limit
    raw_temperature = data.get('temperature') if isinstance(data, dict) else None
    raw_top_p = data.get('top_p') if isinstance(data, dict) else None
    temperature = _clamp_float(raw_temperature, 0.0, 2.0, 0.0)
    top_p = _clamp_float(raw_top_p, 0.0, 1.0, 1.0)
    try:
        raw_model = str(data.get('model') or '').strip()
    except Exception:
        raw_model = ''
    model_value = raw_model or DEFAULT_MODEL
    base_reasoning = _normalize_reasoning_effort_value(
        data.get('reasoning_effort') if isinstance(data, dict) else None,
        ''
    )
    reasoning_value = base_reasoning if _supports_reasoning(model_value) else ''

    settings_payload = data.get('settings') if isinstance(data, dict) else {}
    defaults_input = settings_payload.get('defaults') if isinstance(settings_payload, dict) else {}
    per_model_input = settings_payload.get('per_model') if isinstance(settings_payload, dict) else {}

    defaults_temperature = _clamp_float(
        defaults_input.get('temperature') if isinstance(defaults_input, dict) else None,
        0.0,
        2.0,
        temperature,
    )
    defaults_top_p = _clamp_float(
        defaults_input.get('top_p') if isinstance(defaults_input, dict) else None,
        0.0,
        1.0,
        top_p,
    )
    defaults_reasoning = _normalize_reasoning_effort_value(
        defaults_input.get('reasoning_effort') if isinstance(defaults_input, dict) else None,
        reasoning_value or 'medium',
    )
    defaults_settings = {
        'temperature': defaults_temperature,
        'top_p': defaults_top_p,
        'reasoning_effort': defaults_reasoning,
    }

    per_model_settings: Dict[str, Dict[str, object]] = {}
    if isinstance(per_model_input, dict):
        for model_key_raw, settings in per_model_input.items():
            model_key = str(model_key_raw or '').strip()
            if not model_key:
                continue
            if not isinstance(settings, dict):
                settings = {}
            sanitized_entry: Dict[str, object] = {}
            if _supports_temperature(model_key) and 'temperature' in settings:
                sanitized_entry['temperature'] = _clamp_float(
                    settings.get('temperature'),
                    0.0,
                    2.0,
                    defaults_temperature,
                )
            if _supports_top_p(model_key) and 'top_p' in settings:
                sanitized_entry['top_p'] = _clamp_float(
                    settings.get('top_p'),
                    0.0,
                    1.0,
                    defaults_top_p,
                )
            if _supports_reasoning(model_key) and 'reasoning_effort' in settings:
                sanitized_entry['reasoning_effort'] = _normalize_reasoning_effort_value(
                    settings.get('reasoning_effort'),
                    defaults_reasoning,
                )
            per_model_settings[model_key] = sanitized_entry

    default_entry = per_model_settings.setdefault(model_value, {})
    if _supports_temperature(model_value):
        default_entry['temperature'] = _clamp_float(
            default_entry.get('temperature', temperature),
            0.0,
            2.0,
            temperature,
        )
    else:
        default_entry.pop('temperature', None)
    if _supports_top_p(model_value):
        default_entry['top_p'] = _clamp_float(
            default_entry.get('top_p', top_p),
            0.0,
            1.0,
            top_p,
        )
    else:
        default_entry.pop('top_p', None)
    if _supports_reasoning(model_value):
        default_entry['reasoning_effort'] = _normalize_reasoning_effort_value(
            default_entry.get('reasoning_effort') or reasoning_value or defaults_reasoning,
            defaults_reasoning,
        )
    else:
        default_entry.pop('reasoning_effort', None)

    safe = {
        'name': nm,
        'display_name': display_name,
        'prompt': str(data.get('prompt') or '')[:200000],
        'temperature': default_entry.get('temperature', temperature),
        'top_p': default_entry.get('top_p', top_p),
        'model': model_value,
        'reasoning_effort': default_entry.get('reasoning_effort', '') if _supports_reasoning(model_value) else '',
        'settings': {
            'defaults': defaults_settings,
            'per_model': per_model_settings,
        },
        'updated_at': time.time(),
    }
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.json.tmp')
    try:
        tmp.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(p)
        try:
            p.chmod(0o600)
        except Exception:
            pass
        # return the canonical stored agent name
        return nm
    except Exception:
        try:
            if tmp.exists(): tmp.unlink()
        except Exception:
            pass
        return None

def delete_agent_file(name: str) -> bool:
    nm = safe_agent_name(name)
    if not nm:
        return False
    p = agent_filepath(nm)
    try:
        if p.exists():
            p.unlink()
            return True
    except Exception:
        return False
    return False


def _api_base_to_handle_base(api_base: str) -> str:
    if not api_base:
        return ""
    normalized = api_base.rstrip('/')
    if '/api/' in normalized:
        return normalized.split('/api/', 1)[0]
    if normalized.endswith('/api'):
        return normalized[:-4]
    return normalized


def describe_library(api_base: Optional[str]) -> Dict[str, str]:
    normalized = (api_base or '').rstrip('/')
    if not normalized:
        return {
            'label': '',
            'code': '',
            'api_base': '',
            'handle_base': '',
            'netloc': '',
        }

    override = KNOWN_LIBRARY_OVERRIDES.get(normalized, {})
    handle_base = override.get('handle_base') or _api_base_to_handle_base(normalized)
    parsed = urlparse(handle_base or normalized)
    label = override.get('label') or (parsed.netloc or normalized)
    code = override.get('code') or (parsed.netloc.split('.', 1)[0] if parsed.netloc else '')

    return {
        'label': label,
        'code': code,
        'api_base': normalized,
        'handle_base': handle_base,
        'netloc': parsed.netloc or '',
    }

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ComparisonHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        # Agents API endpoints
        if path == '/agents' or path == '/agents/list':
            items = list_agents_files()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'agents': items}, ensure_ascii=False).encode('utf-8'))
            return
        if path == '/agents/get':
            qs = parse_qs(parsed.query)
            name = (qs.get('name') or [''])[0]
            data = read_agent_file(name)
            if data is None:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'not_found'}).encode('utf-8'))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            return

        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()

            html = '''<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ALTO Processing Comparison</title>
    <style>
:root {
            --thumbnail-drawer-width: clamp(260px, 35vw, 460px);
            --thumbnail-toggle-size: 32px;
        }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .page-shell {
            position: relative;
            max-width: 1200px;
            margin: 0 auto;
            padding-left: 0;
            overflow: visible;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            position: relative;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
        }
        #thumbnail-drawer {
            position: absolute;
            top: 0;
            left: 0;
            width: var(--thumbnail-drawer-width);
            transform: translateX(-100%);
            transition: transform 0.3s ease, opacity 0.3s ease;
            pointer-events: auto;
        }
        .thumbnail-panel {
            background: white;
            border: 1px solid #dbe4f0;
            border-right: none;
            border-radius: 8px 0 0 8px;
            box-shadow: none;
            padding: 16px 16px 16px 18px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            height: 100%;
            max-height: none;
            overflow: hidden;
        }
        body.thumbnail-drawer-collapsed #thumbnail-drawer {
            transform: translateX(0);
            opacity: 0;
            pointer-events: none;
        }
        body.thumbnail-drawer-collapsed .thumbnail-panel {
            pointer-events: none;
        }
        .thumbnail-toggle {
            position: absolute;
            top: 18px;
            left: 0;
            transform: translateX(-50%);
            width: var(--thumbnail-toggle-size);
            height: var(--thumbnail-toggle-size);
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            color: #1f2933;
            font-weight: 600;
            border: 1px solid #d0d7e2;
            border-radius: 10px;
            box-shadow: none;
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease, box-shadow 0.25s ease;
            z-index: 6;
        }
        .thumbnail-toggle:hover {
            background: #1f78ff;
            color: #ffffff;
            box-shadow: none;
        }
        .page-jump {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        .page-jump label {
            margin: 0;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            color: #1f2933;
        }
        #page-number-input {
            width: 72px;
            padding: 6px 8px;
            border: 1px solid #cfd4dc;
            border-radius: 4px;
            font-size: 14px;
            background: #fff;
        }
        #page-number-input:disabled {
            background-color: #f1f3f5;
            color: #98a0ab;
        }
        .page-jump-total {
            color: #52606d;
            font-size: 13px;
        }
        .thumbnail-scroll {
            flex: 1 1 auto;
            overflow-y: auto;
            padding: 8px;
            border: 1px solid #e0e6ef;
            border-radius: 8px;
            background: #f9fbff;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.08);
            max-height: inherit;
            min-height: 0;
        }
        .thumbnail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 8px;
        }
        .page-thumbnail {
            position: relative;
            border: none;
            background: #ffffff;
            color: inherit;
            border-radius: 6px;
            padding: 0;
            cursor: pointer;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.1);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            font-family: inherit;
            aspect-ratio: 3 / 4;
        }
        .thumbnail-placeholder {
            position: absolute;
            inset: 8px;
            border-radius: 6px;
            background: linear-gradient(135deg, #e7edf5 0%, #f1f5fb 100%);
            transition: opacity 0.2s ease;
        }
        .page-thumbnail:hover {
            background: #ffffff;
            transform: translateY(-2px);
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.18);
        }
        .page-thumbnail:disabled {
            cursor: default;
            opacity: 0.6;
            box-shadow: none;
        }
        .page-thumbnail:disabled:hover {
            transform: none;
            box-shadow: none;
        }
        .page-thumbnail img {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #f0f2f7;
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        .page-thumbnail.is-loaded .thumbnail-placeholder {
            opacity: 0;
            visibility: hidden;
        }
        .page-thumbnail.is-loaded img {
            opacity: 1;
        }
        .page-thumbnail-label {
            position: absolute;
            top: 6px;
            right: 6px;
            background: rgba(15, 23, 42, 0.75);
            color: #ffffff;
            font-size: 11px;
            padding: 3px 5px;
            border-radius: 4px;
            line-height: 1;
        }
        .page-thumbnail.is-active {
            outline: 3px solid #1f78ff;
            outline-offset: 0;
            box-shadow: 0 0 0 2px rgba(31, 120, 255, 0.25);
        }
        .page-thumbnail:focus-visible {
            outline: 3px solid #1f78ff;
            outline-offset: 0;
        }
        .thumbnail-empty {
            font-size: 13px;
            color: #5f6b7c;
            padding: 20px 8px;
            text-align: center;
        }
        .main-content {
            min-width: 0;
        }
        @media (max-width: 1100px) {
            body {
                padding: 12px;
            }
            .page-shell {
                max-width: none;
            }
            .container {
                padding: 16px;
                padding-top: 20px;
                padding-bottom: 20px;
            }
            #thumbnail-drawer {
                position: static;
                width: 100%;
                transform: none;
                margin-bottom: 12px;
            }
            body.thumbnail-drawer-collapsed #thumbnail-drawer {
                transform: none;
            }
            .thumbnail-panel {
                max-height: none;
                width: 100%;
                border-right: 1px solid #dbe4f0;
                border-radius: 8px;
                pointer-events: auto;
            }
            .thumbnail-scroll {
                max-height: none;
            }
            .thumbnail-toggle {
                position: static;
                margin-left: auto;
                box-shadow: none;
            }
            body.thumbnail-drawer-collapsed #thumbnail-scroll {
                visibility: visible;
            }
        }
        @media (max-height: 700px) {
            .thumbnail-panel {
                max-height: calc(100vh - 120px);
            }
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            box-sizing: border-box;
        }
        textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            box-sizing: border-box;
            resize: none;
        }
        button:not(.page-thumbnail):not(.thumbnail-toggle):not(.diff-toggle):not(.agent-diff-toggle) {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:not(.page-thumbnail):not(.thumbnail-toggle):not(.diff-toggle):not(.agent-diff-toggle):hover {
            background-color: #0056b3;
        }
        button:not(.page-thumbnail):not(.thumbnail-toggle):not(.diff-toggle):not(.agent-diff-toggle):disabled {
            background-color: #9aa0a6;
            cursor: not-allowed;
        }
        .form-group.uuid-group {
            width: 100%;
        }
        .uuid-input-group {
            display: flex;
            align-items: stretch;
            gap: 8px;
        }
        .inline-tooltip-anchor {
            position: relative;
            display: inline-flex;
            align-items: center;
        }
        .uuid-input-group input {
            flex: 1;
            min-width: 0;
        }
        .uuid-input-group button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #cfd4dc;
            background: #f9fafb;
            color: #4b5563;
            border-radius: 4px;
            padding: 0;
            font-size: 16px;
            line-height: 1;
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease;
            width: 40px;
            height: 38px;
        }
        .uuid-input-group button:hover {
            background: #e5e7eb;
            color: #111827;
        }
        .uuid-input-group button:focus-visible {
            outline: none;
            box-shadow: 0 0 0 2px rgba(31, 120, 255, 0.25);
        }
        .uuid-input-group button:disabled {
            border-color: #cfd4dc;
            color: #9aa0a6;
            background: #f5f6f8;
            cursor: not-allowed;
        }
        .inline-tooltip {
            position: absolute;
            left: 50%;
            top: calc(100% + 6px);
            transform: translate(-50%, 0);
            background: rgba(229, 241, 255, 0.96);
            color: #0b3d8a;
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid rgba(183, 212, 255, 0.9);
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
            opacity: 0;
            pointer-events: none;
            white-space: nowrap;
            transition: opacity 0.15s ease, transform 0.15s ease;
            z-index: 30;
        }
        .inline-tooltip.is-visible {
            opacity: 1;
            transform: translate(-50%, -4px);
        }
        .agent-diff-section {
            margin-top: 16px;
            display: none;
            flex-direction: column;
            gap: 16px;
        }
        .agent-diff-section.is-visible {
            display: flex;
        }
        .agent-diff-controls {
            display: inline-flex;
            align-items: stretch;
            border: 1px solid #cdd5e0;
            border-radius: 999px;
            overflow: hidden;
            background: #f8f9fc;
        }
        .agent-diff-toggle {
            border: none;
            background: transparent;
            padding: 6px 18px;
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease;
        }
        .agent-diff-toggle + .agent-diff-toggle {
            border-left: 1px solid #cdd5e0;
        }
        .agent-diff-toggle.is-active {
            background: #1f78ff;
            color: #ffffff;
        }
        .agent-diff-toggle:focus-visible {
            outline: none;
            box-shadow: inset 0 0 0 2px rgba(31, 120, 255, 0.4);
        }
        .results {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .action-row {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 20px;
            flex-wrap: wrap;
        }
        .tools-row {
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 15px;
            width: 100%;
            justify-content: center;
            position: relative;
        }
        .navigation-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: center;
            margin: 0 auto;
        }
        .navigation-controls button {
            padding: 8px 12px;
        }
        .navigation-controls span {
            min-width: 60px;
            text-align: center;
            font-weight: bold;
            color: #333;
        }
        .page-info-layout {
            display: flex;
            align-items: stretch;
            gap: 16px;
            position: relative;
        }
        .page-details {
            flex: 1 1 auto;
            min-width: 0;
        }
        .page-preview {
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            justify-content: flex-start;
            gap: 8px;
            overflow: visible;
            min-width: 140px;
            margin-left: auto;
            align-self: flex-start;
            height: auto;
            min-height: 120px;
            min-height: 200px;
        }
        .page-preview.preview-visible {
            display: flex;
        }
        .page-preview img {
            border-radius: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2);
            max-height: 100%;
        }
        #preview-image-thumb {
            display: block;
            width: auto;
            height: auto;
            max-width: 100%;
            max-height: 180px;
            object-fit: contain;
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        .preview-error #preview-image-thumb {
            display: none;
        }
        .preview-error #preview-status {
            color: red;
            font-weight: bold;
        }
        .preview-large {
            pointer-events: auto;
            position: absolute;
            top: 0;
            right: 0;
            transform-origin: top right;
            transform: translate(16px, -16px) scale(0.35);
            opacity: 0;
            visibility: hidden;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            z-index: 30;
            transition: transform 0.5s ease, opacity 0.5s ease;
        }
        .page-preview.preview-loaded:hover ~ .preview-large,
        .preview-large.preview-large-visible {
            transform: translate(16px, -16px) scale(1);
            opacity: 1;
            visibility: visible;
        }
        .preview-large img {
            display: block;
            border-radius: 4px;
        }
        #preview-status {
            width: 100%;
            text-align: right;
            max-width: 220px;
        }
        .result-box {
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
        }
        .result-box h3 {
            margin-top: 0;
            color: #333;
        }
        .diff-section {
            margin-top: 28px;
            display: none;
            flex-direction: column;
            gap: 18px;
        }
        .diff-section.is-visible {
            display: flex;
        }
        .diff-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
        }
        .diff-heading {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
            color: #1f2933;
            letter-spacing: -0.01em;
        }
        .diff-controls {
            display: inline-flex;
            align-items: stretch;
            border: 1px solid #cdd5e0;
            border-radius: 999px;
            overflow: hidden;
            background: #f8f9fc;
        }
        .diff-toggle {
            border: none;
            background: transparent;
            padding: 6px 18px;
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease;
        }
        .diff-toggle + .diff-toggle {
            border-left: 1px solid #cdd5e0;
        }
        .diff-toggle.is-active {
            background: #1f78ff;
            color: #ffffff;
        }
        .diff-toggle:focus-visible {
            outline: none;
            box-shadow: inset 0 0 0 2px rgba(31, 120, 255, 0.4);
        }
        .result-rendered {
            line-height: 1.65;
        }
        .result-html-section {
            margin-top: 16px;
        }
        .diff-content {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 16px;
            white-space: normal;
            word-break: break-word;
            overflow-x: auto;
            min-height: 64px;
            font-family: inherit;
            line-height: 1.6;
        }
        .diff-html {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin: 0;
        }
        .diff-block {
            background: #ffffff;
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 10px 12px;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.02);
        }
        .diff-block--added {
            background: rgba(46, 160, 67, 0.12);
            border-color: rgba(46, 160, 67, 0.35);
        }
        .diff-block--removed {
            background: rgba(219, 68, 55, 0.12);
            border-color: rgba(219, 68, 55, 0.35);
        }
        .diff-block--changed {
            background: rgba(255, 196, 0, 0.18);
            border-color: rgba(255, 196, 0, 0.5);
        }
        .diff-added {
            background: rgba(46, 160, 67, 0.38);
            border-radius: 3px;
            padding: 0 2px;
        }
        .diff-removed {
            background: rgba(219, 68, 55, 0.38);
            border-radius: 3px;
            padding: 0 2px;
        }
        .diff-loading {
            color: #52606d;
            font-style: italic;
        }
        .diff-error {
            color: #b91c1c;
            font-weight: 600;
        }
        .diff-placeholder {
            color: #6b7280;
            font-style: italic;
        }
        .loading {
            position: fixed;
            inset: 0;
            background: rgba(255, 255, 255, 0.82);
            backdrop-filter: blur(3px);
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 12px;
            text-align: center;
            z-index: 200;
        }
        .container.is-loading .loading {
            display: flex;
        }
        .loading-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
        }
        .loading-spinner {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 4px solid rgba(0, 123, 255, 0.25);
            border-top-color: #007bff;
            animation: spin 0.8s linear infinite;
        }
        /* small inline spinner used next to dropdowns */
        .inline-spinner {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid rgba(0,0,0,0.08);
            border-top-color: #007bff;
            animation: spin 0.8s linear infinite;
            display: inline-block;
            vertical-align: middle;
        }
        .loading p {
            margin: 0;
            color: #333;
            font-weight: 600;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .error {
            color: red;
            padding: 10px;
            background: #ffe6e6;
            border-radius: 4px;
            margin: 10px 0;
        }
        .success {
            color: green;
            padding: 10px;
            background: #e6ffe6;
            border-radius: 4px;
            margin: 10px 0;
        }
        .info-section {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #fafafa;
            position: relative;
        }

        .info-section h2 {
            margin-top: 0;
            color: #333;
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 8px 16px;
            margin: 0;
        }
        .metadata-grid dt {
            font-weight: bold;
            color: #333;
        }
        .metadata-grid dd {
            margin: 0 0 8px 0;
            color: #555;
        }
        .book-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 8px 12px;
            margin-bottom: 12px;
        }
        .book-chip {
            background: #eef3ff;
            border: 1px solid #d9e2ff;
            border-radius: 6px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .book-chip strong {
            font-size: 12px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #2a3cb5;
        }
        .book-chip-value {
            font-size: 14px;
            color: #1f2933;
        }
        .book-chip-meta {
            font-size: 12px;
            color: #52606d;
        }
        .muted {
            color: #555;
        }
        pre {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgb(0,0,0);
            background-color: rgba(0,0,0,0.4);
        }
        .modal-content {
            background-color: #fefefe;
            margin: auto;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            padding: 0;
            border: 1px solid #888;
            width: 80%;
            max-width: 1200px;
            max-height: 70%;
            overflow-y: auto;
        }
        .modal-header {
            cursor: move;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 20px;
            background-color: #fefefe;
            border-bottom: 1px solid #ddd;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        .modal-content pre {
            padding: 20px;
            margin: 0;
        }
        .close {
            color: #aaa;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover,
        .close:focus {
            color: black;
            text-decoration: none;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="page-shell">
        <aside id="thumbnail-drawer" aria-label="Náhledy stránek">
            <div class="thumbnail-panel">
                <div class="page-jump">
                    <label for="page-number-input">Strana</label>
                    <input type="number" id="page-number-input" min="1" step="1" inputmode="numeric" aria-label="Zadat číslo strany">
                    <span id="page-number-total" class="page-jump-total"></span>
                </div>
                <div id="thumbnail-scroll" class="thumbnail-scroll">
                    <div id="thumbnail-grid" class="thumbnail-grid">
                        <div class="thumbnail-empty">Náhledy budou k dispozici po načtení knihy.</div>
                    </div>
                </div>
            </div>
        </aside>
        <button id="thumbnail-toggle" class="thumbnail-toggle" type="button" aria-expanded="true" aria-controls="thumbnail-grid" aria-label="Skrýt náhledy">&gt;</button>
        <div class="container">
            <div class="main-content">
                <h1>ALTO Processing Comparison</h1>
                <p>Porovnání původního TypeScript a nového Python zpracování ALTO XML</p>

                <div class="form-group uuid-group">
                    <label for="uuid">UUID stránky nebo dokumentu:</label>
                    <div class="uuid-input-group">
                        <input type="text" id="uuid" placeholder="Zadejte UUID (např. dcbff727-d3cd-4e8b-9309-9a8647ddaba5)" value="dcbff727-d3cd-4e8b-9309-9a8647ddaba5" autocomplete="off" autocorrect="off" autocapitalize="none" spellcheck="false">
                        <span class="inline-tooltip-anchor">
                            <button type="button" id="uuid-copy" title="Zkopírovat UUID" aria-label="Zkopírovat UUID">📋</button>
                            <div id="uuid-copy-tooltip" class="inline-tooltip" role="status" aria-live="polite"></div>
                        </span>
                        <span class="inline-tooltip-anchor">
                            <button type="button" id="uuid-paste" title="Vložit UUID ze schránky" aria-label="Vložit UUID ze schránky">📥</button>
                            <div id="uuid-paste-tooltip" class="inline-tooltip" role="status" aria-live="polite"></div>
                        </span>
                        <button type="button" id="uuid-clear" title="Vymazat UUID" aria-label="Vymazat UUID">✕</button>
                    </div>
                </div>

                <div class="action-row">
                    <button id="load-button" type="button" onclick="handleLoadClick()">Načíst stránku</button>
                </div>

                <div id="book-info" class="info-section" style="display: none;">
                    <h2 id="book-title">Informace o knize</h2>
                    <p id="book-handle" class="muted"></p>
                    <p id="book-library" class="muted"></p>
                    <div id="book-constants" class="book-summary-grid" style="display: none;"></div>
                    <div id="book-metadata-empty" class="muted" style="display: none;">Metadata se nepodařilo načíst.</div>
                    <dl id="book-metadata" class="metadata-grid"></dl>
                </div>

                <div id="page-info" class="info-section" style="display: none;">
                    <div class="page-info-layout">
                        <div class="page-details">
                            <h2>Informace o straně</h2>
                            <p id="page-summary" class="muted"></p>
                            <p id="page-side" class="muted"></p>
                            <p id="page-uuid" class="muted"></p>
                            <p id="page-handle" class="muted"></p>
                        </div>
                        <div class="page-alto-btn">
                            <span id="alto-preview-btn" style="display: none;">Zobrazit ALTO</span>
                        </div>
                        <div id="page-preview" class="page-preview" tabindex="0">
                            <div id="preview-status" class="muted"></div>
                            <img id="preview-image-thumb" alt="Náhled stránky">
                        </div>
                        <div id="preview-large" class="preview-large" aria-hidden="true">
                            <img id="preview-image-large" alt="Náhled stránky ve větší velikosti">
                        </div>
                    </div>
                </div>

                <div id="page-tools" class="tools-row" style="display: none;">
                    <div class="navigation-controls">
                        <button id="prev-page" type="button" aria-label="Předchozí stránka">◀</button>
                        <span id="page-position">-</span>
                        <button id="next-page" type="button" aria-label="Další stránka">▶</button>
                    </div>
                </div>
                <div id="results" class="results" style="display: none;">
                    <div class="result-box">
                        <h3>TypeScript výsledek (simulace)</h3>
                        <div id="typescript-result" class="result-rendered"></div>
                    </div>
                    <div class="result-box">
                        <h3>Python výsledek</h3>
                        <div id="python-result" class="result-rendered"></div>
                    </div>
                </div>
                <section id="diff-section" class="diff-section">
                    <div class="diff-header">
                        <h2 class="diff-heading">Porovnání formátovaného HTML</h2>
                        <div id="diff-mode-controls" class="diff-controls" role="group" aria-label="Režim zvýraznění">
                            <button type="button" class="diff-toggle" data-diff-mode="word">Slova</button>
                            <button type="button" class="diff-toggle" data-diff-mode="char">Znaky</button>
                        </div>
                    </div>
                    <div id="html-diff" class="results">
                        <div class="result-box">
                            <h3>Python diff</h3>
                            <div id="python-html" class="diff-html"></div>
                        </div>
                        <div class="result-box">
                            <h3>TypeScript diff</h3>
                            <div id="typescript-html" class="diff-html"></div>
                        </div>
                    </div>
                </section>

                <!-- LLM agent settings UI -->
                <div id="agent-row" class="info-section" style="margin-top:18px;">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <label for="agent-select" style="margin:0;font-weight:600;">Agent:</label>
                        <select id="agent-select" aria-label="Vyberte agenta"></select>
                        <div id="agent-select-spinner" title="Načítám agenty" style="margin-left:8px;display:none;">
                            <span class="inline-spinner" aria-hidden="true"></span>
                        </div>
                        <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                            <label style="display:inline-flex;align-items:center;gap:6px;">
                                <input id="agent-auto-correct" type="checkbox">
                                <span style="font-size:13px;">Automaticky opravovat</span>
                            </label>
                            <button id="agent-run" type="button" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;">Oprav</button>
                            <button id="agent-expand-toggle" type="button" aria-expanded="false" title="Zobrazit nastavení agenta" style="height:36px;display:inline-flex;align-items:center;justify-content:center;padding:6px 10px;">⚙️</button>
                        </div>
                    </div>

                    <div id="agent-settings" style="display:none;margin-top:12px;border-top:1px solid #e6e9ef;padding-top:12px;">
                        <div class="form-group">
                            <label for="agent-name">Název agenta</label>
                            <input type="text" id="agent-name" placeholder="Např. default-editor">
                        </div>
                        <div class="form-group">
                            <label for="agent-prompt">Prompt</label>
                            <textarea id="agent-prompt" rows="6" style="width:100%;font-family:monospace;">Zadejte prompt...</textarea>
                        </div>
                        <div class="form-group">
                            <label for="agent-default-model">Model</label>
                            <select id="agent-default-model" aria-label="Vyberte model pro agenta" style="min-width:170px;"></select>
                        </div>
                        <div id="agent-parameter-fields" class="form-group" style="margin-top:8px;"></div>
                        <div style="display:flex;gap:8px;margin-top:12px;">
                            <span class="inline-tooltip-anchor">
                                <button id="agent-save" type="button">Uložit agenta</button>
                                <div id="agent-save-tooltip" class="inline-tooltip" role="status" aria-live="polite"></div>
                            </span>
                            <button id="agent-delete" type="button" style="background:#e53e3e;">Smazat</button>
                        </div>
                    </div>
                    <div id="agent-output" style="display:none;margin-top:12px;">
                        <div id="agent-output-status" class="muted" style="margin-bottom:6px;"></div>
                        <pre id="agent-output-text" style="white-space:pre-wrap;"></pre>
                    </div>
                    <div id="agent-results" class="results" style="display:none;margin-top:16px;">
                        <div class="result-box">
                            <h3>Agent – původní Python</h3>
                            <div id="agent-result-original" class="result-rendered"></div>
                        </div>
                        <div class="result-box">
                            <h3>Agent – opravený náhled</h3>
                            <div id="agent-result-corrected" class="result-rendered"></div>
                        </div>
                    </div>
                    <div style="margin-top:8px;display:flex;justify-content:flex-end;">
                        <div id="agent-diff-mode-controls" class="agent-diff-controls" role="group" aria-label="Režim zvýraznění agent diffu" style="display:none;">
                            <button type="button" class="agent-diff-toggle" data-diff-mode="word">Slova</button>
                            <button type="button" class="agent-diff-toggle" data-diff-mode="char">Znaky</button>
                        </div>
                    </div>
                </div>
            </div>
            <div id="loading" class="loading" aria-live="polite" aria-hidden="true">
                <div class="loading-content">
                    <div class="loading-spinner" role="presentation"></div>
                    <p>Zpracovávám ALTO data...</p>
                </div>
            </div>
        </div>
    </div>

    <script type="application/json" id="default-agent-model">__DEFAULT_AGENT_MODEL__</script>
    <script type="application/json" id="default-agent-prompt-data">__DEFAULT_AGENT_PROMPT__</script>
    <script>
        let previewObjectUrl = null;
        let previewFetchToken = null;
        let previewImageUuid = null;

        let currentBook = null;
        let currentPage = null;
        let currentLibrary = null;
        let navigationState = null;
        let processRequestToken = 0;

        const pageCache = new Map();
        const inflightProcessRequests = new Map();
        const previewCache = new Map();
        const inflightPreviewRequests = new Map();
        let cacheWindowUuids = new Set();
        let currentAltoXml = "";
        let bookPages = [];
        let lastRenderedBookUuid = null;
        let lastActiveThumbnailUuid = null;
        let thumbnailDrawerCollapsed = false;
        let thumbnailObserver = null;
        const DIFF_MODES = {
            NONE: 'none',
            WORD: 'word',
            CHAR: 'char',
        };
        const DIFF_MODE_STORAGE_KEY = 'altoDiffMode';
        let diffMode = DIFF_MODES.NONE;
        const diffCache = new Map();
        const agentResultCache = new Map();
        const agentFingerprintCache = new Map();
        const AGENT_DIFF_MODES = {
            WORD: 'word',
            CHAR: 'char',
        };
        const AGENT_DIFF_MODE_STORAGE_KEY = 'altoAgentDiffMode';
        let agentDiffMode = AGENT_DIFF_MODES.WORD;
        let lastAgentOriginalHtml = '';
        let lastAgentCorrectedHtml = '';
        let lastAgentCorrectedIsHtml = false;
        let lastAgentOriginalDocumentJson = '';
        let lastAgentCorrectedDocumentJson = '';
        let lastAgentCacheBaseKey = null;
        let agentDiffRequestToken = 0;
        let thumbnailHeightSyncPending = false;
        let currentResults = {
            python: "",
            typescript: "",
            baseKey: "",
        };
        let autoRunScheduled = false;
        const tooltipTimers = new Map();
        let currentAgentName = '';
        let currentAgentDraft = null;
        let agentDraftDirty = false;

const NOTE_STYLE_ATTR = ' style="display:block;font-size:0.82em;color:#1e5aa8;font-weight:bold;"';
        // --- LLM agent management ---
        const AGENTS_STORAGE_KEY = 'altoAgents_v1';
        const AGENT_SELECTED_KEY = 'altoAgentSelected_v1';
        const AGENT_AUTO_CORRECT_KEY = 'altoAgentAutoCorrect_v1';

        // client-side in-memory cache of agents (name -> agent object)
        const agentsCache = {};
        const DEFAULT_AGENT_MODEL = (() => {
            const fallback = 'gpt-4.1-mini';
            const element = document.getElementById('default-agent-model');
            if (!element) {
                return fallback;
            }
            const raw = element.textContent || '';
            try {
                const parsed = JSON.parse(raw);
                return typeof parsed === 'string' && parsed.trim().length ? parsed : fallback;
            } catch (err) {
                console.warn('Nelze načíst výchozí model agenta:', err);
                return raw.trim().length ? raw : fallback;
            }
        })();
        const AVAILABLE_AGENT_MODELS = [
            'gpt-4.1-mini',
            'gpt-4.1',
            'gpt-4o-mini',
            'gpt-4o',
            'gpt-4-0613',
            'gpt-3.5-turbo',
            'gpt-5',
            'o1',
            'o1-mini',
            'o3',
            'o3-mini',
        ];
        const DEFAULT_REASONING_EFFORT = 'medium';
        const DEFAULT_TEMPERATURE = 0.2;
        const DEFAULT_TOP_P = 1.0;
        const REASONING_MODEL_PREFIXES = ['gpt-5', 'o1', 'o3', 'o4'];
        const MODEL_CAPABILITIES_CACHE = new Map();
        const DEFAULT_AGENT_PROMPT = (() => {
            const fallback = 'Jsi pečlivý korektor češtiny. Dostaneš JSON s klíči "language_hint", "page_meta" a "blocks". Blocks je pole objektů {id, type, text}. Oprav překlepy a zjevné OCR chyby pouze v hodnotách "text". Nesjednocuj styl, neměň typy bloků ani jejich pořadí. Zachovej diakritiku, pokud lze. Odpovídej pouze validním JSON se stejnou strukturou a klíči jako vstup.';
            const element = document.getElementById('default-agent-prompt-data');
            if (!element) {
                return fallback;
            }
            const raw = element.textContent || '';
            try {
                const parsed = JSON.parse(raw);
                return typeof parsed === 'string' && parsed.trim().length ? parsed : fallback;
            } catch (err) {
                console.warn('Nelze načíst výchozí prompt agenta:', err);
                return raw.trim().length ? raw : fallback;
            }
        })();
        const DEFAULT_LANGUAGE_HINT = 'cs';

        function normalizeModelId(model) {
            return String(model || '').trim().toLowerCase();
        }

        function isReasoningModel(model) {
            const normalized = normalizeModelId(model);
            if (!normalized) return false;
            for (const prefix of REASONING_MODEL_PREFIXES) {
                const lowered = prefix.toLowerCase();
                if (normalized === lowered || normalized.startsWith(`${lowered}-`)) {
                    return true;
                }
            }
            return false;
        }

        function getModelCapabilities(model) {
            const normalized = normalizeModelId(model);
            if (!normalized) {
                return { temperature: true, top_p: true, reasoning: false };
            }
            if (MODEL_CAPABILITIES_CACHE.has(normalized)) {
                return MODEL_CAPABILITIES_CACHE.get(normalized);
            }
            const capabilities = isReasoningModel(normalized)
                ? { temperature: false, top_p: false, reasoning: true }
                : { temperature: true, top_p: true, reasoning: false };
            MODEL_CAPABILITIES_CACHE.set(normalized, capabilities);
            return capabilities;
        }

        function clamp(value, min, max) {
            if (!Number.isFinite(value)) return min;
            return Math.min(max, Math.max(min, value));
        }

        function clampTemperature(value) {
            return clamp(value, 0, 2);
        }

        function clampTopP(value) {
            return clamp(value, 0, 1);
        }

        function createDefaultModelSettings() {
            return {
                temperature: DEFAULT_TEMPERATURE,
                top_p: DEFAULT_TOP_P,
                reasoning_effort: DEFAULT_REASONING_EFFORT,
            };
        }

        function sanitizeSettingsObject(source) {
            const result = {};
            if (!source || typeof source !== 'object') {
                return result;
            }
            if (Object.prototype.hasOwnProperty.call(source, 'temperature')) {
                const value = Number(source.temperature);
                if (Number.isFinite(value)) {
                    result.temperature = clampTemperature(value);
                }
            }
            if (Object.prototype.hasOwnProperty.call(source, 'top_p')) {
                const value = Number(source.top_p);
                if (Number.isFinite(value)) {
                    result.top_p = clampTopP(value);
                }
            }
            if (Object.prototype.hasOwnProperty.call(source, 'reasoning_effort')) {
                const raw = String(source.reasoning_effort || '').trim().toLowerCase();
                if (['low', 'medium', 'high'].includes(raw)) {
                    result.reasoning_effort = raw;
                }
            }
            return result;
        }

        function normalizeModelSettings(modelId, source, defaults) {
            const capabilities = getModelCapabilities(modelId);
            const sanitizedSource = sanitizeSettingsObject(source || {});
            const sanitizedDefaults = sanitizeSettingsObject(defaults || {});
            const baseDefaults = createDefaultModelSettings();
            const result = {};
            if (capabilities.temperature) {
                if (Object.prototype.hasOwnProperty.call(sanitizedSource, 'temperature')) {
                    result.temperature = clampTemperature(sanitizedSource.temperature);
                } else if (Object.prototype.hasOwnProperty.call(sanitizedDefaults, 'temperature')) {
                    result.temperature = clampTemperature(sanitizedDefaults.temperature);
                } else {
                    result.temperature = baseDefaults.temperature;
                }
            }
            if (capabilities.top_p) {
                if (Object.prototype.hasOwnProperty.call(sanitizedSource, 'top_p')) {
                    result.top_p = clampTopP(sanitizedSource.top_p);
                } else if (Object.prototype.hasOwnProperty.call(sanitizedDefaults, 'top_p')) {
                    result.top_p = clampTopP(sanitizedDefaults.top_p);
                } else {
                    result.top_p = baseDefaults.top_p;
                }
            }
            if (capabilities.reasoning) {
                const sourceValue = sanitizedSource.reasoning_effort || sanitizedDefaults.reasoning_effort || baseDefaults.reasoning_effort;
                result.reasoning_effort = ['low', 'medium', 'high'].includes(sourceValue) ? sourceValue : baseDefaults.reasoning_effort;
            }
            return result;
        }

        function deepCloneAgent(agent) {
            try {
                return JSON.parse(JSON.stringify(agent || {}));
            } catch (err) {
                return agent ? { ...agent } : {};
            }
        }

        function normalizeAgentData(raw) {
            const base = deepCloneAgent(raw || {});
            const name = typeof base.name === 'string' ? base.name.trim() : '';
            const displayName = typeof base.display_name === 'string' && base.display_name.trim().length
                ? base.display_name
                : name;
            const promptValue = typeof base.prompt === 'string' && base.prompt.trim().length
                ? base.prompt
                : DEFAULT_AGENT_PROMPT;
            const modelCandidate = typeof base.model === 'string' && base.model.trim().length
                ? base.model.trim()
                : DEFAULT_AGENT_MODEL;
            const defaults = {
                ...createDefaultModelSettings(),
                ...sanitizeSettingsObject(base.settings && base.settings.defaults),
            };
            const perModel = {};
            if (base.settings && typeof base.settings === 'object' && base.settings.per_model && typeof base.settings.per_model === 'object') {
                for (const [modelIdRaw, settings] of Object.entries(base.settings.per_model)) {
                    const modelId = String(modelIdRaw || '').trim();
                    if (!modelId) continue;
                    perModel[modelId] = normalizeModelSettings(modelId, settings, defaults);
                }
            }
            const legacySettings = sanitizeSettingsObject(base);
            if (Object.keys(legacySettings).length) {
                const current = perModel[modelCandidate] || {};
                perModel[modelCandidate] = normalizeModelSettings(modelCandidate, { ...current, ...legacySettings }, defaults);
            }
            if (!perModel[modelCandidate]) {
                perModel[modelCandidate] = normalizeModelSettings(modelCandidate, {}, defaults);
            }
            return {
                name,
                display_name: displayName || name,
                prompt: promptValue,
                model: modelCandidate,
                settings: {
                    defaults,
                    per_model: perModel,
                },
                updated_at: typeof base.updated_at === 'number' ? base.updated_at : undefined,
            };
        }

        function ensureAgentDraftStructure(draft) {
            if (!draft || typeof draft !== 'object') {
                return null;
            }
            if (!draft.settings || typeof draft.settings !== 'object') {
                draft.settings = { defaults: createDefaultModelSettings(), per_model: {} };
            }
            if (!draft.settings.defaults || typeof draft.settings.defaults !== 'object') {
                draft.settings.defaults = createDefaultModelSettings();
            }
            if (!draft.settings.per_model || typeof draft.settings.per_model !== 'object') {
                draft.settings.per_model = {};
            }
            return draft;
        }

        function ensureAgentDraftModelSettings(draft, modelId) {
            ensureAgentDraftStructure(draft);
            const modelKey = modelId || (draft && draft.model) || DEFAULT_AGENT_MODEL;
            if (!draft.settings.per_model[modelKey]) {
                draft.settings.per_model[modelKey] = normalizeModelSettings(modelKey, {}, draft.settings.defaults);
            }
            return draft.settings.per_model[modelKey];
        }

        function getEffectiveModelSettings(agent, modelId) {
            if (!agent) {
                return normalizeModelSettings(modelId, {}, createDefaultModelSettings());
            }
            const draft = ensureAgentDraftStructure(deepCloneAgent(agent));
            const defaults = draft.settings.defaults || createDefaultModelSettings();
            const perModel = draft.settings.per_model && draft.settings.per_model[modelId]
                ? draft.settings.per_model[modelId]
                : {};
            return normalizeModelSettings(modelId, perModel, defaults);
        }

        function markAgentDirty() {
            agentDraftDirty = true;
        }

        function resetAgentDirty() {
            agentDraftDirty = false;
        }

        function buildAgentSavePayload(draftInput) {
            const normalized = normalizeAgentData(draftInput || currentAgentDraft || {});
            ensureAgentDraftStructure(normalized);
            const modelId = normalized.model || DEFAULT_AGENT_MODEL;
            ensureAgentDraftModelSettings(normalized, modelId);
            const defaultsPayload = {
                ...createDefaultModelSettings(),
                ...sanitizeSettingsObject(normalized.settings.defaults),
            };
            const perModelPayload = {};
            for (const [modelKey, settings] of Object.entries(normalized.settings.per_model || {})) {
                perModelPayload[modelKey] = normalizeModelSettings(modelKey, settings, defaultsPayload);
            }
            const effective = getEffectiveModelSettings(normalized, modelId);
            const capabilities = getModelCapabilities(modelId);
            const payload = {
                name: normalized.name,
                display_name: normalized.display_name || normalized.name,
                prompt: normalized.prompt || '',
                model: modelId,
                settings: {
                    defaults: defaultsPayload,
                    per_model: perModelPayload,
                },
            };
            if (capabilities.temperature && Object.prototype.hasOwnProperty.call(effective, 'temperature')) {
                payload.temperature = effective.temperature;
            }
            if (capabilities.top_p && Object.prototype.hasOwnProperty.call(effective, 'top_p')) {
                payload.top_p = effective.top_p;
            }
            if (capabilities.reasoning && Object.prototype.hasOwnProperty.call(effective, 'reasoning_effort')) {
                payload.reasoning_effort = effective.reasoning_effort;
            }
            return payload;
        }

        function renderAgentModelOptions(selectedModel) {
            const selectEl = document.getElementById('agent-default-model');
            if (!selectEl) {
                return;
            }
            const modelOptions = [...AVAILABLE_AGENT_MODELS];
            if (currentAgentDraft && currentAgentDraft.settings && currentAgentDraft.settings.per_model) {
                Object.keys(currentAgentDraft.settings.per_model).forEach((modelId) => {
                    if (modelId && !modelOptions.includes(modelId)) {
                        modelOptions.push(modelId);
                    }
                });
            }
            if (selectedModel && !modelOptions.includes(selectedModel)) {
                modelOptions.push(selectedModel);
            }
            selectEl.innerHTML = '';
            modelOptions.forEach((modelId) => {
                const option = document.createElement('option');
                option.value = modelId;
                option.textContent = modelId;
                selectEl.appendChild(option);
            });
            const normalized = selectedModel && selectedModel.trim().length ? selectedModel : DEFAULT_AGENT_MODEL;
            selectEl.value = normalized;
        }

        function renderAgentParameterControls(modelId) {
            const container = document.getElementById('agent-parameter-fields');
            if (!container) {
                return;
            }
            if (!currentAgentDraft) {
                container.innerHTML = '<div class="muted">Nejprve vyberte agenta.</div>';
                return;
            }
            const capabilities = getModelCapabilities(modelId);
            const modelSettings = ensureAgentDraftModelSettings(currentAgentDraft, modelId);
            const controls = [];
            if (capabilities.temperature) {
                controls.push(`
                    <div class="agent-param" data-param="temperature" style="flex:1;min-width:220px;">
                        <label for="agent-param-temperature" style="display:block;margin-bottom:4px;font-weight:600;">Temperature</label>
                        <div style="display:flex;gap:8px;align-items:center;">
                            <input id="agent-param-temperature" type="range" min="0" max="2" step="0.05" style="flex:1;">
                            <input id="agent-param-temperature-number" type="number" min="0" max="2" step="0.05" style="width:80px;">
                        </div>
                    </div>
                `);
            }
            if (capabilities.top_p) {
                controls.push(`
                    <div class="agent-param" data-param="top_p" style="flex:1;min-width:220px;">
                        <label for="agent-param-top-p" style="display:block;margin-bottom:4px;font-weight:600;">Top P</label>
                        <div style="display:flex;gap:8px;align-items:center;">
                            <input id="agent-param-top-p" type="range" min="0" max="1" step="0.01" style="flex:1;">
                            <input id="agent-param-top-p-number" type="number" min="0" max="1" step="0.01" style="width:80px;">
                        </div>
                    </div>
                `);
            }
            if (capabilities.reasoning) {
                controls.push(`
                    <div class="agent-param" data-param="reasoning_effort" style="flex:1;min-width:220px;">
                        <label for="agent-param-reasoning" style="display:block;margin-bottom:4px;font-weight:600;">Reasoning</label>
                        <select id="agent-param-reasoning" style="min-width:140px;">
                            <option value="low">Low</option>
                            <option value="medium">Medium</option>
                            <option value="high">High</option>
                        </select>
                    </div>
                `);
            }
            if (!controls.length) {
                container.innerHTML = '<div class="muted">Tento model nemá nastavitelné parametry.</div>';
                return;
            }
            container.innerHTML = `<div style="display:flex;gap:12px;flex-wrap:wrap;">${controls.join('')}</div>`;
            const tempRange = document.getElementById('agent-param-temperature');
            const tempNumber = document.getElementById('agent-param-temperature-number');
            const topRange = document.getElementById('agent-param-top-p');
            const topNumber = document.getElementById('agent-param-top-p-number');
            const reasoningSelect = document.getElementById('agent-param-reasoning');

            if (tempRange && tempNumber) {
                const value = Object.prototype.hasOwnProperty.call(modelSettings, 'temperature')
                    ? modelSettings.temperature
                    : DEFAULT_TEMPERATURE;
                tempRange.value = value;
                tempNumber.value = value;
                const updateTemperature = (nextValue) => {
                    const parsed = clampTemperature(Number(nextValue));
                    tempRange.value = parsed;
                    tempNumber.value = parsed;
                    modelSettings.temperature = parsed;
                    markAgentDirty();
                };
                tempRange.addEventListener('input', () => updateTemperature(tempRange.value));
                tempNumber.addEventListener('change', () => updateTemperature(tempNumber.value));
            }

            if (topRange && topNumber) {
                const value = Object.prototype.hasOwnProperty.call(modelSettings, 'top_p')
                    ? modelSettings.top_p
                    : DEFAULT_TOP_P;
                topRange.value = value;
                topNumber.value = value;
                const updateTopP = (nextValue) => {
                    const parsed = clampTopP(Number(nextValue));
                    topRange.value = parsed;
                    topNumber.value = parsed;
                    modelSettings.top_p = parsed;
                    markAgentDirty();
                };
                topRange.addEventListener('input', () => updateTopP(topRange.value));
                topNumber.addEventListener('change', () => updateTopP(topNumber.value));
            }

            if (reasoningSelect) {
                const raw = String(modelSettings.reasoning_effort || '').trim().toLowerCase();
                const normalized = ['low', 'medium', 'high'].includes(raw) ? raw : DEFAULT_REASONING_EFFORT;
                reasoningSelect.value = normalized;
                reasoningSelect.addEventListener('change', () => {
                    const candidate = String(reasoningSelect.value || '').trim().toLowerCase();
                    const next = ['low', 'medium', 'high'].includes(candidate) ? candidate : DEFAULT_REASONING_EFFORT;
                    reasoningSelect.value = next;
                    modelSettings.reasoning_effort = next;
                    markAgentDirty();
                });
            }
        }

        async function loadAgents(options = {}) {
            const force = options && typeof options === 'object' && options.force === true;
            const keys = Object.keys(agentsCache);
            if (keys.length && !force) {
                const out = {};
                keys.sort().forEach(k => { out[k] = { name: k, display_name: agentsCache[k].display_name || k }; });
                return out;
            }
            try {
                const res = await fetch('/agents/list', { cache: 'no-store' });
                if (!res.ok) return {};
                const data = await res.json();
                const out = {};
                // server returns list items which may include parsed agent object
                const fetchedAgents = [];
                (data.agents || []).forEach(a => {
                    const baseAgent = a && a.agent && typeof a.agent === 'object'
                        ? { ...deepCloneAgent(a.agent), name: a.name, display_name: a.display_name || a.name }
                        : { name: a.name, display_name: a.display_name || a.name, prompt: DEFAULT_AGENT_PROMPT, model: DEFAULT_AGENT_MODEL };
                    const normalized = normalizeAgentData(baseAgent);
                    agentsCache[a.name] = normalized;
                    cacheAgentFingerprint(a.name, normalized);
                    out[a.name] = { name: a.name, display_name: normalized.display_name || a.name };
                    fetchedAgents.push(a.name);
                });
                console.info('[AgentDebug] loadAgents ->', fetchedAgents);
                if (force) {
                    Object.keys(agentsCache).forEach((name) => {
                        if (!fetchedAgents.includes(name)) {
                            delete agentsCache[name];
                            agentFingerprintCache.delete(name);
                        }
                    });
                }
                return out;
            } catch (err) {
                console.warn('Nelze načíst agenty ze serveru, fallback na empty:', err);
                return {};
            }
        }

        // saveAgents is no longer used client-side; server persists agents

        function persistSelectedAgent(name) {
            try { localStorage.setItem(AGENT_SELECTED_KEY, name || ''); } catch (e) {}
        }

        function loadSelectedAgent() {
            try { return localStorage.getItem(AGENT_SELECTED_KEY) || ''; } catch (e) { return ''; }
        }

        function persistAutoCorrect(value) {
            try { localStorage.setItem(AGENT_AUTO_CORRECT_KEY, value ? '1' : '0'); } catch (e) {}
        }

        function loadAutoCorrect() {
            try { return localStorage.getItem(AGENT_AUTO_CORRECT_KEY) === '1'; } catch (e) { return false; }
        }

        function cacheAgentFingerprint(name, agent) {
            if (!name) {
                return null;
            }
            if (agent && typeof agent === 'object') {
                const payload = extractAgentFingerprintPayload(agent);
                const serialized = JSON.stringify(payload);
                const hash = computeSimpleHash(serialized);
                const record = { hash, serialized };
                agentFingerprintCache.set(name, record);
                return record;
            }
            return agentFingerprintCache.get(name) || null;
        }

        function extractAgentFingerprintPayload(agent) {
            if (!agent || typeof agent !== 'object') {
                return null;
            }
            const normalized = normalizeAgentData(agent);
            ensureAgentDraftStructure(normalized);
            const modelId = normalized.model || DEFAULT_AGENT_MODEL;
            const defaults = {
                ...createDefaultModelSettings(),
                ...sanitizeSettingsObject(normalized.settings.defaults),
            };
            const perModelEntries = Object.entries(normalized.settings.per_model || {})
                .map(([modelKey, settings]) => [modelKey, normalizeModelSettings(modelKey, settings, defaults)])
                .sort((a, b) => a[0].localeCompare(b[0]));
            const effective = getEffectiveModelSettings(normalized, modelId);
            return {
                prompt: String(normalized.prompt || ''),
                model: modelId,
                defaults,
                per_model: perModelEntries,
                effective,
            };
        }

        function computeAgentFingerprint(name) {
            if (!name) {
                return '';
            }
            const agent = (currentAgentName === name && currentAgentDraft)
                ? currentAgentDraft
                : agentsCache[name];
            const record = cacheAgentFingerprint(name, agent);
            if (record && typeof record.hash === 'number') {
                return `${name}:${record.hash}`;
            }
            const cached = agentFingerprintCache.get(name);
            if (cached && typeof cached.hash === 'number') {
                return `${name}:${cached.hash}`;
            }
            return name;
        }

        async function renderAgentSelector(force = false) {
            const select = document.getElementById('agent-select');
            const spinnerWrapper = document.getElementById('agent-select-spinner');
            if (spinnerWrapper) spinnerWrapper.style.display = 'inline-block';
            if (!select) {
                if (spinnerWrapper) spinnerWrapper.style.display = 'none';
                return;
            }
            select.innerHTML = '';
            const agents = await loadAgents({ force });
            const selected = loadSelectedAgent();
            const names = Object.keys(agents).sort();
            // If no agents on server, create some defaults client-side and save
            if (!names.length) {
                const defaults = ['editor-default','cleanup-a','semantic-fix'];
                for (const n of defaults) {
                    const draftAgent = {
                        name: n,
                        display_name: n,
                        prompt: DEFAULT_AGENT_PROMPT,
                        model: DEFAULT_AGENT_MODEL,
                        settings: {
                            defaults: createDefaultModelSettings(),
                            per_model: {
                                [DEFAULT_AGENT_MODEL]: {
                                    temperature: DEFAULT_TEMPERATURE,
                                    top_p: DEFAULT_TOP_P,
                                },
                            },
                        },
                    };
                    const payload = buildAgentSavePayload(draftAgent);
                    await fetch('/agents/save', {
                        method: 'POST', headers: {'Content-Type':'application/json'},
                        body: JSON.stringify(payload),
                    });
                }
                if (spinnerWrapper) spinnerWrapper.style.display = 'none';
                return renderAgentSelector(force);
            }

            // show nicer labels when display_name exists
            for (const name of names) {
                const option = document.createElement('option');
                option.value = name;
                // prefer display_name from cache
                option.textContent = (agents[name] && agents[name].display_name) ? agents[name].display_name : name;
                select.appendChild(option);
            }

            if (selected && names.includes(selected)) {
                select.value = selected;
            } else {
                select.selectedIndex = 0;
                persistSelectedAgent(select.value);
            }
            if (spinnerWrapper) spinnerWrapper.style.display = 'none';
            // update the UI fields for the selected agent from cache (avoid extra fetch)
            try {
                const sel = select.value;
                if (sel && agentsCache[sel]) {
                    setAgentFields(agentsCache[sel]);
                    persistSelectedAgent(sel);
                } else {
                    await updateAgentUIFromSelection();
                }
            } catch (e) {
                await updateAgentUIFromSelection();
            }
        }

        async function updateAgentUIFromSelection() {
            const select = document.getElementById('agent-select');
            if (!select) return;
            const name = select.value;
            const auto = document.getElementById('agent-auto-correct');
            if (auto) auto.checked = loadAutoCorrect();
            const fallbackAgent = normalizeAgentData({
                name,
                display_name: name,
                prompt: DEFAULT_AGENT_PROMPT,
                model: DEFAULT_AGENT_MODEL,
            });
            try {
                // prefer in-memory cache where possible
                if (agentsCache[name]) {
                    setAgentFields(deepCloneAgent(agentsCache[name]));
                    persistSelectedAgent(name);
                    return;
                }
                const res = await fetch(`/agents/get?name=${encodeURIComponent(name)}`, { cache: 'no-store' });
                if (!res.ok) {
                    setAgentFields(fallbackAgent);
                    persistSelectedAgent('');
                    return;
                }
                const agent = await res.json();
                const normalized = normalizeAgentData({ ...(agent || {}), name });
                agentsCache[name] = normalized;
                setAgentFields(normalized);
                persistSelectedAgent(name);
            } catch (err) {
                console.warn('Nelze načíst agenta:', err);
                setAgentFields(fallbackAgent);
            }
        }

        function autoResizeTextarea(textarea) {
            if (!textarea) return;
            if (!textarea.dataset.minHeight) {
                const computed = window.getComputedStyle(textarea);
                const lineHeight = parseFloat(computed.lineHeight) || 0;
                const rows = parseInt(textarea.getAttribute('rows') || '0', 10);
                const paddingTop = parseFloat(computed.paddingTop) || 0;
                const paddingBottom = parseFloat(computed.paddingBottom) || 0;
                const borderTop = parseFloat(computed.borderTopWidth) || 0;
                const borderBottom = parseFloat(computed.borderBottomWidth) || 0;
                const estimated = rows && lineHeight ? (rows * lineHeight) + paddingTop + paddingBottom + borderTop + borderBottom : textarea.clientHeight;
                textarea.dataset.minHeight = String(estimated || 0);
            }
            const minHeight = parseFloat(textarea.dataset.minHeight || '0') || 0;
            textarea.style.height = 'auto';
            const newHeight = textarea.scrollHeight;
            if (newHeight && newHeight > 0) {
                textarea.style.height = `${Math.max(newHeight, minHeight)}px`;
            } else if (minHeight) {
                textarea.style.height = `${minHeight}px`;
            }
            delete textarea.dataset.needsResize;
            scheduleThumbnailDrawerHeightSync();
        }

        function setAgentFields(agent) {
            const normalized = normalizeAgentData(agent || {});
            currentAgentDraft = deepCloneAgent(normalized);
            currentAgentName = normalized.name || '';
            ensureAgentDraftStructure(currentAgentDraft);
            const nameEl = document.getElementById('agent-name');
            const promptEl = document.getElementById('agent-prompt');
            const modelEl = document.getElementById('agent-default-model');
            if (nameEl) {
                nameEl.value = currentAgentDraft.name || '';
            }
            if (promptEl) {
                promptEl.value = currentAgentDraft.prompt || DEFAULT_AGENT_PROMPT;
                if (!promptEl.dataset.autoresizeBound) {
                    promptEl.dataset.autoresizeBound = 'true';
                    promptEl.addEventListener('input', () => autoResizeTextarea(promptEl));
                }
                if (promptEl.offsetParent !== null) {
                    requestAnimationFrame(() => autoResizeTextarea(promptEl));
                } else {
                    promptEl.dataset.needsResize = 'true';
                }
            }
            if (modelEl) {
                renderAgentModelOptions(currentAgentDraft.model);
            }
            renderAgentParameterControls(currentAgentDraft.model || DEFAULT_AGENT_MODEL);
            resetAgentDirty();
        }

        async function saveCurrentAgent() {
            const nameEl = document.getElementById('agent-name');
            const promptEl = document.getElementById('agent-prompt');
            const modelEl = document.getElementById('agent-default-model');
            if (!nameEl) return;
            if (!currentAgentDraft) {
                currentAgentDraft = normalizeAgentData({
                    name: '',
                    prompt: promptEl ? promptEl.value : DEFAULT_AGENT_PROMPT,
                    model: modelEl ? modelEl.value : DEFAULT_AGENT_MODEL,
                });
            }
            const draft = ensureAgentDraftStructure(currentAgentDraft);
            const previousName = currentAgentName || draft.name || '';
            draft.name = String(nameEl.value || '').trim();
            if (!draft.name) {
                alert('Agent musí mít název');
                return;
            }
            draft.display_name = draft.display_name || draft.name;
            draft.prompt = promptEl ? promptEl.value : '';
            const selectedModel = modelEl ? String(modelEl.value || '').trim() : DEFAULT_AGENT_MODEL;
            draft.model = selectedModel || DEFAULT_AGENT_MODEL;
            ensureAgentDraftModelSettings(draft, draft.model);

            const payload = buildAgentSavePayload(draft);
            try {
                const res = await fetch('/agents/save', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
                if (!res.ok) {
                    const txt = await res.text().catch(()=>'');
                    throw new Error('save failed ' + txt);
                }
                const data = await res.json().catch(() => null);
                const stored = (data && data.stored_name) ? data.stored_name : draft.name;
                const agentResponse = (data && data.agent) ? data.agent : payload;
                const normalized = normalizeAgentData({ ...(agentResponse || {}), name: stored });

                // update caches
                agentsCache[stored] = normalized;
                cacheAgentFingerprint(stored, normalized);
                if (previousName && stored !== previousName) {
                    delete agentsCache[previousName];
                    agentFingerprintCache.delete(previousName);
                }

                persistSelectedAgent(stored);
                currentAgentName = stored;
                currentAgentDraft = deepCloneAgent(normalized);
                await renderAgentSelector(true);
                setAgentFields(normalized);
                resetAgentDirty();
                showTooltip('agent-save-tooltip', 'Agent uložen');
            } catch (err) {
                alert('Nelze uložit agenta: ' + err);
                showTooltip('agent-save-tooltip', 'Uložení selhalo', 2600);
            }
        }

        async function deleteCurrentAgent() {
            const select = document.getElementById('agent-select');
            if (!select) return;
            const name = select.value;
            if (!name) return;
            if (!confirm(`Opravdu chcete agenta "${name}" smazat?`)) return;
            try {
                const res = await fetch('/agents/delete', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
                if (!res.ok) throw new Error('delete failed');
                delete agentsCache[name];
                agentFingerprintCache.delete(name);
                persistSelectedAgent('');
                await renderAgentSelector(true);
            } catch (err) {
                alert('Nelze smazat agenta: ' + err);
            }
        }

        // newAgent removed — creation/editing handled via saveCurrentAgent and changing name

        function parseAgentResultDocument(text) {
            if (!text || typeof text !== 'string') {
                return null;
            }
            const trimmed = text.trim();
            if (!trimmed) {
                return null;
            }
            try {
                return JSON.parse(trimmed);
            } catch (err) {
                return null;
            }
        }

        function documentBlocksToHtml(documentPayload) {
            if (!documentPayload || typeof documentPayload !== 'object' || !Array.isArray(documentPayload.blocks)) {
                return null;
            }
            const parts = [];
            for (const block of documentPayload.blocks) {
                if (!block || typeof block.text !== 'string') {
                    continue;
                }
                const text = block.text.trim();
                if (!text) {
                    continue;
                }
                const type = (block.type || '').toLowerCase();
                let tag = 'p';
                let attrs = '';
                let markup = '';
                switch (type) {
                    case 'h1':
                    case 'h2':
                    case 'h3':
                        tag = type;
                        break;
                    case 'small':
                        markup = `<p><small>${escapeHtml(text)}</small></p>`;
                        break;
                    case 'note':
                        tag = 'note';
                        attrs = NOTE_STYLE_ATTR;
                        break;
                    case 'centered':
                        tag = 'div';
                        attrs = ' class="centered"';
                        break;
                    case 'blockquote':
                        tag = 'blockquote';
                        break;
                    case 'li':
                        tag = 'p';
                        attrs = ' data-block-type="li"';
                        break;
                    default:
                        tag = 'p';
                        if (type && type !== 'p') {
                            attrs = ` data-block-type="${escapeHtml(type)}"`;
                        }
                        break;
                }
                if (!markup) {
                    if (tag === 'note') {
                        markup = `<br><${tag}${attrs}>${escapeHtml(text)}</${tag}>`;
                    } else {
                        markup = `<${tag}${attrs}>${escapeHtml(text)}</${tag}>`;
                    }
                }
                parts.push(markup);
            }
            return parts.length ? parts.join("") : null;
        }



        function clearAgentOutput() {
            const container = document.getElementById('agent-output');
            const statusEl = document.getElementById('agent-output-status');
            const textEl = document.getElementById('agent-output-text');
            if (container) container.style.display = 'none';
            if (statusEl) {
                statusEl.textContent = '';
                statusEl.style.color = '';
                statusEl.style.fontWeight = '';
            }
            if (textEl) textEl.textContent = '';
            setAgentResultPanels(null, null, false);
        }

        function setAgentOutput(statusText, bodyText, state) {
            const container = document.getElementById('agent-output');
            const statusEl = document.getElementById('agent-output-status');
            const textEl = document.getElementById('agent-output-text');
            if (!container || !statusEl || !textEl) {
                return;
            }

            const normalizedStatus = statusText ? String(statusText).trim() : '';
            const normalizedBody = bodyText ? String(bodyText).trim() : '';
            if (!normalizedStatus && !normalizedBody) {
                container.style.display = 'none';
                statusEl.textContent = '';
                statusEl.style.color = '';
                statusEl.style.fontWeight = '';
                textEl.textContent = '';
                scheduleThumbnailDrawerHeightSync();
                return;
            }

            container.style.display = 'block';
            statusEl.textContent = normalizedStatus;
            if (state === 'error') {
                statusEl.style.color = '#b91c1c';
                statusEl.style.fontWeight = '';
            } else if (state === 'pending') {
                statusEl.style.color = '';
                statusEl.style.fontWeight = '600';
            } else {
                statusEl.style.color = '';
                statusEl.style.fontWeight = '';
            }
            textEl.textContent = normalizedBody;
            scheduleThumbnailDrawerHeightSync();
        }

        function formatHtmlForPre(html) {
            if (!html) {
                return '';
            }
            return escapeHtml(html || '');
        }

        function setAgentResultPanels(originalHtml, correctedContent, correctedIsHtml) {
            lastAgentOriginalHtml = originalHtml || '';
            lastAgentCorrectedHtml = correctedContent || '';
            lastAgentCorrectedIsHtml = Boolean(correctedIsHtml);
            if (!originalHtml || !correctedContent) {
                lastAgentCacheBaseKey = null;
            }
            if (!originalHtml) {
                lastAgentOriginalDocumentJson = '';
            }
            if (!correctedContent) {
                lastAgentCorrectedDocumentJson = '';
            }
            const container = document.getElementById('agent-results');
            const originalEl = document.getElementById('agent-result-original');
            const correctedEl = document.getElementById('agent-result-corrected');
            if (!container || !originalEl || !correctedEl) {
                return;
            }

            console.group('[AgentDebug] Agent result panels');
            console.debug('[AgentDebug] Original Python HTML length:', originalHtml ? originalHtml.length : 0);
            console.debug('[AgentDebug] Original Python HTML preview:', originalHtml);
            console.debug('[AgentDebug] Corrected content length:', correctedContent ? correctedContent.length : 0, 'isHtml:', correctedIsHtml);
            console.debug('[AgentDebug] Corrected content preview:', correctedContent);
            console.groupEnd();

            if (!originalHtml && !correctedContent) {
                container.style.display = 'none';
                originalEl.innerHTML = '';
                correctedEl.innerHTML = '';
                renderAgentDiff(null, agentDiffMode, { hidden: true });
                scheduleThumbnailDrawerHeightSync();
                return;
            }

            container.style.display = 'grid';

            if (originalHtml) {
                originalEl.innerHTML = `<pre>${originalHtml}</pre>`;
            } else {
                originalEl.innerHTML = '<div class="muted">Žádná data.</div>';
            }

            if (correctedContent) {
                correctedEl.innerHTML = correctedIsHtml
                    ? `<pre>${correctedContent}</pre>`
                    : `<pre>${escapeHtml(correctedContent)}</pre>`;
            } else {
                correctedEl.innerHTML = '<div class="muted">Agent nevrátil HTML náhled.</div>';
            }

            const shouldHideDiff = !(originalHtml && correctedContent);
            renderAgentDiff(null, agentDiffMode, { hidden: shouldHideDiff });
            scheduleThumbnailDrawerHeightSync();
        }

        function loadStoredAgentDiffMode() {
            try {
                const stored = localStorage.getItem(AGENT_DIFF_MODE_STORAGE_KEY);
                if (stored === AGENT_DIFF_MODES.WORD || stored === AGENT_DIFF_MODES.CHAR) {
                    return stored;
                }
            } catch (error) {
                console.warn('Nelze načíst uložený režim agent diffu:', error);
            }
            return AGENT_DIFF_MODES.WORD;
        }

        function persistAgentDiffMode(mode) {
            try {
                if (mode === AGENT_DIFF_MODES.WORD || mode === AGENT_DIFF_MODES.CHAR) {
                    localStorage.setItem(AGENT_DIFF_MODE_STORAGE_KEY, mode);
                }
            } catch (error) {
                console.warn('Nelze uložit režim agent diffu:', error);
            }
        }

        function updateAgentDiffToggleState() {
            const container = document.getElementById('agent-diff-mode-controls');
            if (!container) {
                return;
            }
            const buttons = container.querySelectorAll('.agent-diff-toggle');
            buttons.forEach((button) => {
                if (!(button instanceof HTMLElement)) {
                    return;
                }
                const mode = button.getAttribute('data-diff-mode');
                const isActive = mode === agentDiffMode;
                button.classList.toggle('is-active', Boolean(isActive));
                button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
            });
        }

        function renderAgentDiff(diff, mode, options) {
            const originalEl = document.getElementById('agent-result-original');
            const correctedEl = document.getElementById('agent-result-corrected');
            const controls = document.getElementById('agent-diff-mode-controls');
            updateAgentDiffToggleState();
            if (!originalEl || !correctedEl) {
                return;
            }
            const shouldHide = Boolean(options && options.hidden);
            if (!diff || shouldHide) {
                if (controls) {
                    controls.style.display = 'none';
                }
                scheduleThumbnailDrawerHeightSync();
                return;
            }
            const originalPre = originalEl.querySelector('pre');
            const correctedPre = correctedEl.querySelector('pre');
            if (diff.original && originalPre) {
                originalPre.innerHTML = diff.original;
            } else if (diff.original && !originalPre) {
                originalEl.innerHTML = `<pre>${diff.original}</pre>`;
            }
            if (diff.corrected && correctedPre) {
                correctedPre.innerHTML = diff.corrected;
            } else if (diff.corrected && !correctedPre) {
                correctedEl.innerHTML = `<pre>${diff.corrected}</pre>`;
            }
            if (controls) {
                controls.style.display = 'inline-flex';
            }
            scheduleThumbnailDrawerHeightSync();
        }

        async function fetchAgentDiff(originalHtml, correctedHtml, mode) {
            if (!originalHtml || !correctedHtml) {
                return null;
            }
            try {
                const response = await fetch('/agents/diff', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        original: originalHtml,
                        corrected: correctedHtml,
                        mode: mode || agentDiffMode || AGENT_DIFF_MODES.WORD,
                    }),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok || !data || data.ok === false) {
                    const message = data && data.error ? data.error : response.statusText || 'Neznámá chyba';
                    throw new Error(message);
                }
                return data;
            } catch (error) {
                console.warn('Agent diff request selhal:', error);
                return null;
            }
        }

        function computeAgentCacheBaseKey() {
            if (lastAgentCacheBaseKey) {
                return lastAgentCacheBaseKey;
            }
            const select = document.getElementById('agent-select');
            const name = select && select.value ? String(select.value) : '';
            if (!name || !lastAgentOriginalHtml) {
                return null;
            }
            const fingerprint = computeAgentFingerprint(name);
            if (!fingerprint) {
                return null;
            }
            return `${fingerprint}:${computeSimpleHash(lastAgentOriginalHtml)}`;
        }

        async function requestAgentDiff(originalHtml, correctedHtml, correctedIsHtml, cacheBaseKey) {
            const originalPayload = lastAgentOriginalDocumentJson || originalHtml;
            const correctedPayload = lastAgentCorrectedDocumentJson || correctedHtml;
            if (!originalPayload || !correctedPayload) {
                renderAgentDiff(null, agentDiffMode, { hidden: true });
                return null;
            }
            const baseKey = cacheBaseKey || computeAgentCacheBaseKey();
            if (baseKey) {
                lastAgentCacheBaseKey = baseKey;
            }
            agentDiffRequestToken += 1;
            const requestId = agentDiffRequestToken;
            const payload = await fetchAgentDiff(originalPayload, correctedPayload, agentDiffMode);
            if (requestId !== agentDiffRequestToken) {
                return payload;
            }
            const diffData = payload && payload.diff ? payload.diff : null;
            if (diffData) {
                renderAgentDiff(diffData, agentDiffMode);
            } else {
                renderAgentDiff(null, agentDiffMode, { hidden: true });
            }
            if (baseKey && diffData) {
                const cacheKey = `${baseKey}:${agentDiffMode}`;
                agentResultCache.set(cacheKey, {
                    correctedContent: correctedHtml,
                    correctedIsHtml,
                    diff: diffData,
                    mode: agentDiffMode,
                    originalDocumentJson: lastAgentOriginalDocumentJson,
                    correctedDocumentJson: lastAgentCorrectedDocumentJson,
                });
            }
            return payload;
        }

        async function refreshAgentDiff() {
            updateAgentDiffToggleState();
            if (!lastAgentOriginalHtml || !lastAgentCorrectedHtml) {
                renderAgentDiff(null, agentDiffMode, { hidden: true });
                return;
            }
            const baseKey = computeAgentCacheBaseKey();
            if (baseKey) {
                const cacheKey = `${baseKey}:${agentDiffMode}`;
                const cached = agentResultCache.get(cacheKey);
                if (cached && Object.prototype.hasOwnProperty.call(cached, 'diff')) {
                    if (cached.originalDocumentJson) {
                        lastAgentOriginalDocumentJson = cached.originalDocumentJson;
                    }
                    if (cached.correctedDocumentJson) {
                        lastAgentCorrectedDocumentJson = cached.correctedDocumentJson;
                    }
                    if (cached.diff) {
                        renderAgentDiff(cached.diff, cached.mode || agentDiffMode);
                        return;
                    }
                    renderAgentDiff(null, agentDiffMode, { hidden: true });
                    return;
                }
            }
            renderAgentDiff(null, agentDiffMode, { hidden: true });
            await requestAgentDiff(lastAgentOriginalHtml, lastAgentCorrectedHtml, lastAgentCorrectedIsHtml, baseKey);
        }

        function setAgentDiffMode(newMode) {
            const normalized = newMode === AGENT_DIFF_MODES.CHAR ? AGENT_DIFF_MODES.CHAR : AGENT_DIFF_MODES.WORD;
            if (agentDiffMode === normalized) {
                updateAgentDiffToggleState();
                return;
            }
            agentDiffMode = normalized;
            persistAgentDiffMode(agentDiffMode);
            updateAgentDiffToggleState();
            refreshAgentDiff().catch((error) => {
                console.warn('Agent diff přepočet selhal:', error);
            });
        }

        function initializeAgentDiffControls() {
            agentDiffMode = loadStoredAgentDiffMode();
            const container = document.getElementById('agent-diff-mode-controls');
            if (container) {
                container.addEventListener('click', (event) => {
                    const target = event.target;
                    if (!target || !(target instanceof HTMLElement)) {
                        return;
                    }
                    if (!target.matches('.agent-diff-toggle')) {
                        return;
                    }
                    const requestedMode = target.getAttribute('data-diff-mode');
                    if (!requestedMode) {
                        return;
                    }
                    setAgentDiffMode(requestedMode);
                });
            }
            updateAgentDiffToggleState();
        }

        async function runSelectedAgent(options = {}) {
            autoRunScheduled = false;
            const select = document.getElementById('agent-select');
            if (!select) return;
            const name = select.value;
            if (!name) {
                alert('Žádný agent není vybrán');
                return;
            }
            const isAutoInvocation = Boolean(options && options.autoTriggered);
            const pythonHtml = currentResults && currentResults.python ? String(currentResults.python) : '';
            if (!pythonHtml.trim()) {
                alert('Python výstup je prázdný – nejprve načtěte stránku.');
                return;
            }
            const auto = document.getElementById('agent-auto-correct');
            const willAuto = auto && auto.checked;
            persistAutoCorrect(Boolean(willAuto));
            const originalPythonHtml = pythonHtml;

            let workingDraft = null;
            if (currentAgentDraft && currentAgentName === name) {
                workingDraft = currentAgentDraft;
            } else if (agentsCache[name]) {
                workingDraft = deepCloneAgent(agentsCache[name]);
                currentAgentDraft = deepCloneAgent(workingDraft);
                currentAgentName = name;
            } else {
                workingDraft = normalizeAgentData({
                    name,
                    prompt: DEFAULT_AGENT_PROMPT,
                    model: DEFAULT_AGENT_MODEL,
                });
                currentAgentDraft = deepCloneAgent(workingDraft);
                currentAgentName = name;
            }
            ensureAgentDraftStructure(workingDraft);
            const normalizedModel = (workingDraft.model && String(workingDraft.model).trim()) || DEFAULT_AGENT_MODEL;
            ensureAgentDraftModelSettings(workingDraft, normalizedModel);
            const capabilities = getModelCapabilities(normalizedModel);
            const effectiveSettings = getEffectiveModelSettings(workingDraft, normalizedModel);
            const normalizedReasoning = capabilities.reasoning && effectiveSettings.reasoning_effort
                ? effectiveSettings.reasoning_effort
                : DEFAULT_REASONING_EFFORT;

            const pageMeta = {};
            if (currentPage) {
                if (currentPage.pageNumber) pageMeta.page = currentPage.pageNumber;
                else if (typeof currentPage.index === 'number' && !Number.isNaN(currentPage.index)) pageMeta.page = currentPage.index + 1;
                if (typeof currentPage.index === 'number' && !Number.isNaN(currentPage.index)) pageMeta.page_index = currentPage.index;
                if (currentPage.uuid) pageMeta.page_uuid = currentPage.uuid;
                if (currentPage.pageType) pageMeta.page_type = currentPage.pageType;
                if (currentPage.pageSide) pageMeta.page_side = currentPage.pageSide;
            }
            if (currentBook) {
                if (currentBook.title) pageMeta.work = currentBook.title;
                if (currentBook.uuid) pageMeta.book_uuid = currentBook.uuid;
            }

            const payload = {
                name,
                auto_correct: Boolean(willAuto),
                python_html: pythonHtml,
                language_hint: DEFAULT_LANGUAGE_HINT,
                page_meta: pageMeta,
                page_uuid: currentPage && currentPage.uuid ? currentPage.uuid : '',
                book_uuid: currentBook && currentBook.uuid ? currentBook.uuid : '',
                book_title: currentBook && currentBook.title ? currentBook.title : '',
                page_number: currentPage && currentPage.pageNumber ? currentPage.pageNumber : '',
                page_index: currentPage && typeof currentPage.index === 'number' ? currentPage.index : null,
                model: normalizedModel,
                model_override: normalizedModel,
            };
            if (capabilities.temperature && Object.prototype.hasOwnProperty.call(effectiveSettings, 'temperature')) {
                payload.temperature = effectiveSettings.temperature;
            }
            if (capabilities.top_p && Object.prototype.hasOwnProperty.call(effectiveSettings, 'top_p')) {
                payload.top_p = effectiveSettings.top_p;
            }
            if (capabilities.reasoning && Object.prototype.hasOwnProperty.call(effectiveSettings, 'reasoning_effort')) {
                payload.reasoning_effort = effectiveSettings.reasoning_effort;
            }
            payload.agent_snapshot = buildAgentSavePayload(workingDraft);

            const runBtn = document.getElementById('agent-run');
            const originalLabel = runBtn ? runBtn.textContent : '';

            cacheAgentFingerprint(name, workingDraft);
            const agentFingerprint = computeAgentFingerprint(name);
            const pythonHash = computeSimpleHash(pythonHtml);
            const cacheBaseKey = agentFingerprint ? `${agentFingerprint}:${pythonHash}` : null;
            const cacheKey = cacheBaseKey ? `${cacheBaseKey}:${agentDiffMode}` : null;
            lastAgentCacheBaseKey = cacheBaseKey;
            lastAgentOriginalHtml = originalPythonHtml;
            lastAgentCorrectedHtml = '';
            lastAgentCorrectedIsHtml = false;
            lastAgentOriginalDocumentJson = '';
            lastAgentCorrectedDocumentJson = '';
            const cachedResult = cacheKey ? agentResultCache.get(cacheKey) : null;

            if (cachedResult && isAutoInvocation) {
                if (cachedResult.originalDocumentJson) {
                    lastAgentOriginalDocumentJson = cachedResult.originalDocumentJson;
                }
                if (cachedResult.correctedDocumentJson) {
                    lastAgentCorrectedDocumentJson = cachedResult.correctedDocumentJson;
                }
                setAgentResultPanels(originalPythonHtml, cachedResult.correctedContent, cachedResult.correctedIsHtml);
                lastAgentCacheBaseKey = cacheBaseKey;
                if (cachedResult.diff) {
                    renderAgentDiff(cachedResult.diff, cachedResult.mode || agentDiffMode);
                } else if (lastAgentOriginalHtml && lastAgentCorrectedHtml) {
                    await refreshAgentDiff();
                } else {
                    renderAgentDiff(null, agentDiffMode, { hidden: true });
                }
                console.info(`[AgentDebug] Použit cache výsledek pro agenta ${name}`);
                setAgentOutput('', '', 'success');
                return;
            }

            let startTime = null;
            try {
                if (runBtn) {
                    runBtn.disabled = true;
                    runBtn.textContent = 'Spouštím...';
                }
                startTime = performance.now();
                setAgentOutput(`Spouštím agenta ${name}...`, '', 'pending');
                const agentConfig = workingDraft || {};
                console.groupCollapsed(`[AgentDebug] Request → ${name}`);
                console.log('Agent config:', agentConfig);
                console.log('Payload:', payload);
                console.groupEnd();
                const response = await fetch('/agents/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok || !data || data.ok === false) {
                    const message = data && data.error ? data.error : response.statusText || 'Neznámá chyba';
                    throw new Error(message);
                }
                const result = data.result || {};
                const text = typeof result.text === 'string' ? result.text.trim() : '';
                const statusParts = [];
                console.groupCollapsed(`[AgentDebug] Response → ${name}`);
                console.log('Výsledek:', result);
                console.groupEnd();

                const parsedDoc = parseAgentResultDocument(text);
                const originalDocument = result && typeof result.input_document === 'object' ? result.input_document : null;
                if (originalDocument) {
                    try {
                        lastAgentOriginalDocumentJson = JSON.stringify(originalDocument);
                    } catch (jsonError) {
                        lastAgentOriginalDocumentJson = '';
                    }
                } else {
                    lastAgentOriginalDocumentJson = '';
                }
                if (parsedDoc) {
                    try {
                        lastAgentCorrectedDocumentJson = JSON.stringify(parsedDoc);
                    } catch (jsonError) {
                        lastAgentCorrectedDocumentJson = '';
                    }
                } else {
                    lastAgentCorrectedDocumentJson = '';
                }
                const htmlFromDoc = parsedDoc ? documentBlocksToHtml(parsedDoc) : null;
                const autoRequested = Boolean(data.auto_correct);
                if (autoRequested && !htmlFromDoc) {
                    statusParts.push('Výsledek nelze aplikovat (očekáván JSON se strukturou blocks)');
                }

                const correctedContent = htmlFromDoc || (text || '');
                const correctedIsHtml = Boolean(htmlFromDoc);
                setAgentResultPanels(originalPythonHtml, correctedContent, correctedIsHtml);

                if (cacheKey) {
                    agentResultCache.set(cacheKey, {
                        correctedContent,
                        correctedIsHtml,
                        mode: agentDiffMode,
                        originalDocumentJson: lastAgentOriginalDocumentJson,
                        correctedDocumentJson: lastAgentCorrectedDocumentJson,
                    });
                }

                if (correctedContent && originalPythonHtml) {
                    await requestAgentDiff(originalPythonHtml, correctedContent, correctedIsHtml, cacheBaseKey);
                } else {
                    renderAgentDiff(null, agentDiffMode, { hidden: true });
                }

                const elapsedMs = performance.now() - startTime;
                const elapsedLabel = `${(elapsedMs / 1000).toFixed(2)} s`;

                if (statusParts.length) {
                    const statusText = statusParts.join(' · ');
                    if (statusText.toLowerCase().includes('nelze')) {
                        setAgentOutput(statusText, `Hotovo za ${elapsedLabel}`, 'error');
                    } else {
                        setAgentOutput(`Hotovo za ${elapsedLabel}`, statusText, 'success');
                    }
                } else {
                    setAgentOutput(`Hotovo za ${elapsedLabel}`, '', 'success');
                }
            } catch (err) {
                const elapsedMs = startTime !== null && typeof performance !== 'undefined'
                    ? performance.now() - startTime
                    : 0;
                const elapsedLabel = elapsedMs ? ` · ${(elapsedMs / 1000).toFixed(2)} s` : '';
                const message = err && err.message ? err.message : String(err);
                setAgentOutput(`Chyba agenta: ${message}${elapsedLabel}`, '', 'error');
                renderAgentDiff(null, agentDiffMode, { hidden: true });
                console.groupCollapsed(`[AgentDebug] Chyba → ${name}`);
                console.error(err);
                console.groupEnd();
            } finally {
                if (runBtn) {
                    runBtn.disabled = false;
                    runBtn.textContent = originalLabel || 'Oprav';
                }
            }
        }


        async function initializeAgentUI() {
            // Attach listeners immediately so UI is responsive while agent list loads
            const select = document.getElementById('agent-select');
            const runBtn = document.getElementById('agent-run');
            const auto = document.getElementById('agent-auto-correct');
            const toggle = document.getElementById('agent-expand-toggle');
            const settings = document.getElementById('agent-settings');
            const saveBtn = document.getElementById('agent-save');
            const deleteBtn = document.getElementById('agent-delete');
            const modelSelect = document.getElementById('agent-default-model');
            const nameInput = document.getElementById('agent-name');
            const promptTextarea = document.getElementById('agent-prompt');

            if (select) select.addEventListener('change', updateAgentUIFromSelection);
            if (runBtn) runBtn.addEventListener('click', () => runSelectedAgent({ autoTriggered: false }));
            if (auto) {
                auto.checked = loadAutoCorrect();
                auto.addEventListener('change', () => {
                    persistAutoCorrect(auto.checked);
                    if (auto.checked) {
                        autoRunScheduled = false;
                        const selectEl = document.getElementById('agent-select');
                        const runButton = document.getElementById('agent-run');
                        const hasAgent = Boolean(selectEl && selectEl.value);
                        const hasPython = currentResults && typeof currentResults.python === 'string' && currentResults.python.trim().length > 0;
                        if (hasAgent && hasPython && !(runButton && runButton.disabled)) {
                            runSelectedAgent({ autoTriggered: true }).catch((error) => {
                                console.warn('Automatické spuštění agenta po změně zaškrtávacího políčka selhalo:', error);
                            });
                        }
                    } else {
                        autoRunScheduled = false;
                    }
                });
            }
            if (toggle && settings) {
                toggle.addEventListener('click', async () => {
                    const visible = settings.style.display !== 'none';
                    const nowVisible = !visible;
                    settings.style.display = visible ? 'none' : 'block';
                    toggle.setAttribute('aria-expanded', (nowVisible).toString());
                    if (nowVisible) {
                        try { await updateAgentUIFromSelection(); } catch (e) { /* ignore */ }
                        const promptField = document.getElementById('agent-prompt');
                        if (promptField) {
                            requestAnimationFrame(() => autoResizeTextarea(promptField));
                        }
                    }
                    scheduleThumbnailDrawerHeightSync();
                    setTimeout(() => scheduleThumbnailDrawerHeightSync(), 80);
                });
            }
            if (saveBtn) saveBtn.addEventListener('click', async () => { await saveCurrentAgent(); scheduleThumbnailDrawerHeightSync(); });
            if (deleteBtn) deleteBtn.addEventListener('click', async () => { await deleteCurrentAgent(); scheduleThumbnailDrawerHeightSync(); });

            if (nameInput) {
                nameInput.addEventListener('input', () => {
                    if (!currentAgentDraft) {
                        return;
                    }
                    currentAgentDraft.name = String(nameInput.value || '').trim();
                    markAgentDirty();
                });
            }

            if (promptTextarea) {
                promptTextarea.addEventListener('input', () => {
                    if (!currentAgentDraft) {
                        return;
                    }
                    currentAgentDraft.prompt = promptTextarea.value;
                    markAgentDirty();
                    promptTextarea.dataset.needsResize = 'true';
                    autoResizeTextarea(promptTextarea);
                });
            }

            if (modelSelect) {
                modelSelect.addEventListener('change', () => {
                    if (!currentAgentDraft) {
                        return;
                    }
                    const selectedModel = String(modelSelect.value || '').trim() || DEFAULT_AGENT_MODEL;
                    currentAgentDraft.model = selectedModel;
                    ensureAgentDraftModelSettings(currentAgentDraft, selectedModel);
                    renderAgentParameterControls(selectedModel);
                    markAgentDirty();
                    scheduleThumbnailDrawerHeightSync();
                });
            }

            // Load selector asynchronously; don't block UI interactivity
            renderAgentSelector().catch(() => {});
        }
        function createThumbnailLoadManager(concurrency = 6) {
            const queue = [];
            let active = 0;

            function insertTask(task) {
                if (!queue.length) {
                    queue.push(task);
                    return;
                }
                const index = queue.findIndex(item => item.priority > task.priority || (item.priority === task.priority && item.created > task.created));
                if (index === -1) {
                    queue.push(task);
                } else {
                    queue.splice(index, 0, task);
                }
            }

            function startTask(task) {
                const img = task && task.img;
                if (!img || !img.dataset || !img.dataset.src) {
                    if (img && img.dataset) {
                        delete img.dataset.queueId;
                        delete img.dataset.loading;
                    }
                    if (active > 0) {
                        active -= 1;
                    }
                    schedule();
                    return;
                }

                delete img.dataset.queueId;
                if (img.dataset.loaded === 'true' || img.dataset.loading === 'true') {
                    if (active > 0) {
                        active -= 1;
                    }
                    schedule();
                    return;
                }

                active += 1;
                img.dataset.loading = 'true';
                img.src = img.dataset.src;
                if (img.complete) {
                    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                        img.dispatchEvent(new Event('load'));
                    } else if (img.naturalWidth === 0) {
                        img.dispatchEvent(new Event('error'));
                    }
                }
            }

            function schedule() {
                while (active < concurrency && queue.length) {
                    const task = queue.shift();
                    if (!task) {
                        continue;
                    }
                    startTask(task);
                }
            }

            function enqueue(img, priority = 1) {
                if (!img || !img.dataset || !img.dataset.src) {
                    return;
                }
                if (img.dataset.loaded === 'true' || img.dataset.loading === 'true') {
                    return;
                }

                const normalizedPriority = Number.isFinite(priority) ? priority : 1;

                if (normalizedPriority <= 0 && active < concurrency) {
                    startTask({ img, priority: normalizedPriority, created: performance.now() });
                    return;
                }

                if (img.dataset.queueId) {
                    return;
                }

                const task = {
                    img,
                    priority: normalizedPriority,
                    created: performance.now(),
                };
                img.dataset.queueId = String(task.created);
                insertTask(task);
                schedule();
            }

            function markComplete(img, success) {
                if (img && img.dataset) {
                    delete img.dataset.loading;
                    delete img.dataset.queueId;
                    if (success) {
                        img.dataset.loaded = 'true';
                    } else {
                        delete img.dataset.loaded;
                    }
                }
                if (active > 0) {
                    active -= 1;
                }
                schedule();
            }

            function reset() {
                while (queue.length) {
                    const task = queue.shift();
                    if (task && task.img && task.img.dataset) {
                        delete task.img.dataset.queueId;
                        delete task.img.dataset.loading;
                    }
                }
                active = 0;
            }

            return {
                enqueue,
                markComplete,
                reset,
            };
        }

        const thumbnailLoader = createThumbnailLoadManager(6);

        function finalizeThumbnail(img, success) {
            thumbnailLoader.markComplete(img, success);
        }

        function cacheProcessData(uuid, payload) {
            if (!uuid || !payload) {
                return;
            }
            pageCache.set(uuid, {
                payload,
                timestamp: Date.now(),
            });
        }

        async function ensureProcessData(uuid) {
            if (!uuid) {
                return null;
            }
            const cached = pageCache.get(uuid);
            if (cached) {
                cached.timestamp = Date.now();
                return cached.payload;
            }
            if (inflightProcessRequests.has(uuid)) {
                return inflightProcessRequests.get(uuid);
            }

            const promise = (async () => {
                // When available, forward the currently selected library api_base so the server
                // queries the same Kramerius instance the UI is showing. This prevents
                // thumbnail clicks from switching the source library unexpectedly.
                let processUrl = `/process?uuid=${encodeURIComponent(uuid)}`;
                try {
                    if (currentLibrary && currentLibrary.api_base) {
                        processUrl += `&api_base=${encodeURIComponent(currentLibrary.api_base)}`;
                    }
                } catch (err) {
                    // ignore and fall back to default
                }
                const response = await fetch(processUrl, { cache: "no-store" });
                const data = await response.json();
                if (!response.ok || data.error) {
                    const message = data && data.error ? data.error : response.statusText || `HTTP ${response.status}`;
                    throw new Error(message);
                }
                cacheProcessData(uuid, data);
                return data;
            })().finally(() => {
                inflightProcessRequests.delete(uuid);
            });

            inflightProcessRequests.set(uuid, promise);
            return promise;
        }

        function releasePreviewEntry(entry) {
            if (entry && entry.objectUrl && entry.objectUrl !== previewObjectUrl) {
                URL.revokeObjectURL(entry.objectUrl);
            }
        }

        function storePreviewEntry(uuid, result) {
            if (!uuid || !result) {
                return null;
            }
            const existing = previewCache.get(uuid);
            if (existing) {
                releasePreviewEntry(existing);
            }
            const objectUrl = result.objectUrl || URL.createObjectURL(result.blob);
            const payload = {
                blob: result.blob,
                stream: result.stream,
                contentType: result.contentType,
                objectUrl,
                timestamp: Date.now(),
            };
            previewCache.set(uuid, payload);
            return payload;
        }

        async function ensurePreviewEntry(uuid) {
            if (!uuid) {
                return null;
            }
            const cached = previewCache.get(uuid);
            if (cached) {
                cached.timestamp = Date.now();
                return cached;
            }
            if (inflightPreviewRequests.has(uuid)) {
                return inflightPreviewRequests.get(uuid);
            }

            const promise = (async () => {
                const result = await fetchPreviewImage(uuid);
                return storePreviewEntry(uuid, result);
            })().finally(() => {
                inflightPreviewRequests.delete(uuid);
            });

            inflightPreviewRequests.set(uuid, promise);
            return promise;
        }

        function computeCacheWindow(currentUuid, navigation) {
            const target = new Set();
            if (currentUuid) {
                target.add(currentUuid);
            }
            if (navigation) {
                if (navigation.prevUuid) {
                    target.add(navigation.prevUuid);
                }
                if (navigation.nextUuid) {
                    target.add(navigation.nextUuid);
                }
            }
            return target;
        }

        function updateCacheWindow(currentUuid, navigation) {
            cacheWindowUuids = computeCacheWindow(currentUuid, navigation);

            for (const key of Array.from(pageCache.keys())) {
                if (!cacheWindowUuids.has(key)) {
                    pageCache.delete(key);
                }
            }

            for (const [key, entry] of Array.from(previewCache.entries())) {
                if (!cacheWindowUuids.has(key)) {
                    releasePreviewEntry(entry);
                    previewCache.delete(key);
                }
            }
        }

        function schedulePrefetch(navigation) {
            if (!navigation) {
                return;
            }
            const candidates = [navigation.prevUuid, navigation.nextUuid].filter(Boolean);
            candidates.forEach(uuid => {
                prefetchProcess(uuid);
                prefetchPreview(uuid);
            });
        }

        async function prefetchProcess(uuid) {
            if (!uuid || pageCache.has(uuid) || inflightProcessRequests.has(uuid)) {
                return;
            }
            try {
                await ensureProcessData(uuid);
            } catch (error) {
                console.warn("Nepodařilo se přednačíst stránku", uuid, error);
            }
        }

        async function prefetchPreview(uuid) {
            if (!uuid || previewCache.has(uuid) || inflightPreviewRequests.has(uuid)) {
                return;
            }
            try {
                await ensurePreviewEntry(uuid);
            } catch (error) {
                console.warn("Nepodařilo se přednačíst náhled", uuid, error);
            }
        }

        function setThumbnailDrawerCollapsed(collapsed) {
            const desiredState = Boolean(collapsed);
            thumbnailDrawerCollapsed = desiredState;
            document.body.classList.toggle('thumbnail-drawer-collapsed', thumbnailDrawerCollapsed);
            const toggle = document.getElementById('thumbnail-toggle');
            if (toggle) {
                toggle.textContent = thumbnailDrawerCollapsed ? '<' : '>';
                toggle.setAttribute('aria-expanded', (!thumbnailDrawerCollapsed).toString());
                toggle.setAttribute('aria-label', thumbnailDrawerCollapsed ? 'Zobrazit náhledy' : 'Skrýt náhledy');
            }
            const drawerPanel = document.querySelector('#thumbnail-drawer .thumbnail-panel');
            if (drawerPanel) {
                drawerPanel.setAttribute('aria-hidden', thumbnailDrawerCollapsed ? 'true' : 'false');
            }
            const drawer = document.getElementById('thumbnail-drawer');
            if (drawer) {
                drawer.setAttribute('aria-hidden', thumbnailDrawerCollapsed ? 'true' : 'false');
            }
            if (thumbnailDrawerCollapsed) {
                resetThumbnailObserver();
            } else {
                ensureThumbnailObserver();
            }
            if (!thumbnailDrawerCollapsed) {
                scheduleThumbnailDrawerHeightSync(true);
            }
        }

        function navigateToUuid(targetUuid) {
            if (!targetUuid) {
                return;
            }
            const uuidField = document.getElementById("uuid");
            if (uuidField) {
                uuidField.value = targetUuid;
                setUuidButtonsState();
            }
            processAlto();
        }

        function clearThumbnailGrid(message) {
            const grid = document.getElementById("thumbnail-grid");
            if (!grid) {
                return;
            }
            resetThumbnailObserver();
            grid.innerHTML = "";
            if (message) {
                const placeholder = document.createElement("div");
                placeholder.className = "thumbnail-empty";
                placeholder.textContent = message;
                grid.appendChild(placeholder);
            }
        }

        function normalizePages(pages) {
            if (!Array.isArray(pages)) {
                return [];
            }
            return pages.map((entry, idx) => {
                const normalized = Object.assign({}, entry || {});
                if (typeof normalized.index !== "number" || !Number.isFinite(normalized.index)) {
                    normalized.index = idx;
                }
                normalized.uuid = normalized.uuid || "";
                return normalized;
            });
        }

        function updateThumbnailLabels(pages) {
            const grid = document.getElementById("thumbnail-grid");
            if (!grid || !pages.length) {
                return;
            }
            const buttons = grid.querySelectorAll('.page-thumbnail');
            buttons.forEach(button => {
                const listIndex = Number.parseInt(button.dataset.listIndex || "-1", 10);
                const page = Number.isFinite(listIndex) && listIndex >= 0 && listIndex < pages.length ? pages[listIndex] : null;
                const pageNumber = page && page.pageNumber ? page.pageNumber : "";
                const displayIndex = page && typeof page.index === "number" && Number.isFinite(page.index) ? page.index : listIndex;
                const labelText = pageNumber ? `Strana ${pageNumber}` : `Strana ${displayIndex + 1}`;

                button.setAttribute('aria-label', labelText);

                const thumbImage = button.querySelector('img');
                if (thumbImage) {
                    thumbImage.alt = labelText;
                }

                let labelEl = button.querySelector('.page-thumbnail-label');
                if (pageNumber) {
                    if (!labelEl) {
                        labelEl = document.createElement('span');
                        labelEl.className = 'page-thumbnail-label';
                        button.appendChild(labelEl);
                    }
                    labelEl.textContent = pageNumber;
                } else if (labelEl) {
                    labelEl.remove();
                }
            });
        }

        function highlightActiveThumbnail(uuid) {
            const grid = document.getElementById("thumbnail-grid");
            if (!grid) {
                lastActiveThumbnailUuid = null;
                return;
            }

            let activeButton = null;
            grid.querySelectorAll('.page-thumbnail').forEach(button => {
                if (uuid && button.dataset.uuid === uuid) {
                    button.classList.add('is-active');
                    activeButton = button;
                } else {
                    button.classList.remove('is-active');
                }
            });

            const shouldScroll = uuid && uuid !== lastActiveThumbnailUuid && !thumbnailDrawerCollapsed;
            if (activeButton && shouldScroll) {
                const scrollContainer = document.getElementById('thumbnail-scroll');
                if (scrollContainer) {
                    const containerRect = scrollContainer.getBoundingClientRect();
                    const buttonRect = activeButton.getBoundingClientRect();
                    if (buttonRect.top < containerRect.top || buttonRect.bottom > containerRect.bottom) {
                        activeButton.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
                    }
                } else {
                    activeButton.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
                }
            }

            lastActiveThumbnailUuid = uuid || null;
        }

        function scheduleThumbnailDrawerHeightSync(immediate = false) {
            if (immediate) {
                thumbnailHeightSyncPending = false;
                try {
                    syncThumbnailDrawerHeight();
                } catch (error) {
                    console.warn('Thumbnail height sync failed:', error);
                }
                return;
            }
            if (thumbnailHeightSyncPending) {
                return;
            }
            thumbnailHeightSyncPending = true;
            requestAnimationFrame(() => {
                thumbnailHeightSyncPending = false;
                try {
                    syncThumbnailDrawerHeight();
                } catch (error) {
                    console.warn('Thumbnail height sync failed:', error);
                }
            });
        }

        function syncThumbnailDrawerHeight() {
            const drawer = document.getElementById('thumbnail-drawer');
            const panel = drawer ? drawer.querySelector('.thumbnail-panel') : null;
            const container = document.querySelector('.container');
            if (!drawer || !container) {
                return;
            }
            const rect = container.getBoundingClientRect();
            const height = rect && Number.isFinite(rect.height) ? rect.height : container.offsetHeight;
            if (!Number.isFinite(height) || height <= 0) {
                return;
            }
            drawer.style.height = `${height}px`;
            if (panel) {
                const styles = window.getComputedStyle(panel);
                const paddingTop = parseFloat(styles.paddingTop) || 0;
                const paddingBottom = parseFloat(styles.paddingBottom) || 0;
                const borderTop = parseFloat(styles.borderTopWidth) || 0;
                const borderBottom = parseFloat(styles.borderBottomWidth) || 0;
                const innerHeight = Math.max(height - paddingTop - paddingBottom - borderTop - borderBottom, 0);
                panel.style.height = `${innerHeight}px`;
                panel.style.maxHeight = `${innerHeight}px`;
            }
        }

        function resetThumbnailObserver() {
            if (thumbnailObserver) {
                thumbnailObserver.disconnect();
                thumbnailObserver = null;
            }
            thumbnailLoader.reset();
        }

        function ensureThumbnailObserver() {
            if (thumbnailObserver) {
                return thumbnailObserver;
            }
            if (!('IntersectionObserver' in window)) {
                return null;
            }
            const scrollContainer = document.getElementById('thumbnail-scroll');
            if (!scrollContainer) {
                return null;
            }
            thumbnailObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        loadThumbnailImage(img, 1);
                        if (thumbnailObserver) {
                            thumbnailObserver.unobserve(img);
                        }
                    }
                });
            }, {
                root: scrollContainer,
                rootMargin: '200px 0px',
                threshold: 0.05,
            });
            const grid = document.getElementById('thumbnail-grid');
            if (grid) {
                grid.querySelectorAll('img[data-src]').forEach(img => {
                    if (!img.dataset.loaded || img.dataset.loaded !== 'true') {
                        thumbnailObserver.observe(img);
                    }
                });
            }
            return thumbnailObserver;
        }

        function loadThumbnailImage(img, priority = 1) {
            if (!img || !img.dataset || !img.dataset.src) {
                return;
            }
            if (img.dataset.loaded === 'true' || img.dataset.loading === 'true') {
                return;
            }
            let normalizedPriority;
            if (typeof priority === 'boolean') {
                normalizedPriority = priority ? 0 : 1;
            } else if (Number.isFinite(priority)) {
                normalizedPriority = priority;
            } else {
                normalizedPriority = 1;
            }
            thumbnailLoader.enqueue(img, normalizedPriority);
        }

        function updatePageNumberInput() {
            const input = document.getElementById('page-number-input');
            const totalLabel = document.getElementById('page-number-total');
            const total = bookPages.length || (navigationState && typeof navigationState.total === 'number' ? navigationState.total : 0);

            if (totalLabel) {
                totalLabel.textContent = total ? `/ ${total}` : "";
            }

            if (!input) {
                return;
            }

            if (currentPage && typeof currentPage.index === 'number' && Number.isFinite(currentPage.index) && currentPage.index >= 0) {
                input.value = String(currentPage.index + 1);
            } else if (total) {
                input.value = '1';
            } else {
                input.value = "";
            }

            input.disabled = total === 0;
        }

        function ensureThumbnailGrid(pages, bookUuid) {
            const grid = document.getElementById('thumbnail-grid');
            const scrollContainer = document.getElementById('thumbnail-scroll');

            const normalizedPages = normalizePages(pages);
            bookPages = normalizedPages;

            if (!grid) {
                lastRenderedBookUuid = null;
                highlightActiveThumbnail(null);
                updatePageNumberInput();
                return;
            }

            if (!normalizedPages.length) {
                lastRenderedBookUuid = null;
                clearThumbnailGrid('Náhledy nejsou k dispozici.');
                highlightActiveThumbnail(null);
                updatePageNumberInput();
                return;
            }

            const normalizedBookUuid = bookUuid || null;
            const isSameBook = normalizedBookUuid === lastRenderedBookUuid;
            const shouldRender = !isSameBook || grid.querySelectorAll('.page-thumbnail').length !== normalizedPages.length;

            if (shouldRender) {
                const previousScrollTop = isSameBook && scrollContainer ? scrollContainer.scrollTop : 0;
                resetThumbnailObserver();
                grid.innerHTML = "";

                const priorityIndices = new Set();
                for (let i = 0; i < 6 && i < normalizedPages.length; i += 1) {
                    priorityIndices.add(i);
                }
                if (currentPage && typeof currentPage.index === 'number' && Number.isFinite(currentPage.index)) {
                    const base = Math.max(Math.min(currentPage.index - 1, normalizedPages.length - 1), 0);
                    for (let i = base; i <= Math.min(base + 2, normalizedPages.length - 1); i += 1) {
                        priorityIndices.add(i);
                    }
                }

                const observer = ensureThumbnailObserver();

                normalizedPages.forEach((page, listIndex) => {
                    const button = document.createElement('button');
                    button.type = 'button';
                    button.className = 'page-thumbnail';
                    button.dataset.uuid = page.uuid || "";
                    button.dataset.index = String(page.index);
                    button.dataset.listIndex = String(listIndex);

                    const labelText = page.pageNumber ? `Strana ${page.pageNumber}` : `Strana ${page.index + 1}`;
                    button.setAttribute('aria-label', labelText);

                    if (page.uuid) {
                        button.addEventListener('click', () => navigateToUuid(page.uuid));
                    } else {
                        button.disabled = true;
                    }

                    const placeholder = document.createElement('div');
                    placeholder.className = 'thumbnail-placeholder';
                    button.appendChild(placeholder);

                    const img = document.createElement('img');
                    img.loading = 'lazy';
                    img.decoding = 'async';
                    img.alt = labelText;
                    const normalizedUuid = typeof page.uuid === 'string' ? page.uuid : "";
                    const providedThumb = typeof page.thumbnail === 'string' ? page.thumbnail : "";
                    const fallbackThumb = normalizedUuid ? `/preview?uuid=${encodeURIComponent(normalizedUuid)}&stream=IMG_THUMB` : "";
                    const thumbSrc = providedThumb || fallbackThumb;
                    if (thumbSrc) {
                        img.dataset.src = thumbSrc;
                    }
                    const markLoaded = () => {
                        button.classList.add('is-loaded');
                        img.dataset.loaded = 'true';
                    };
                    img.addEventListener('load', () => {
                        const success = img.naturalWidth > 0 && img.naturalHeight > 0;
                        if (success) {
                            markLoaded();
                        } else {
                            button.classList.remove('is-loaded');
                        }
                        finalizeThumbnail(img, success);
                    });
                    img.addEventListener('error', () => {
                        button.classList.remove('is-loaded');
                        finalizeThumbnail(img, false);
                    });
                    button.appendChild(img);

                    if (page.pageNumber) {
                        const badge = document.createElement('span');
                        badge.className = 'page-thumbnail-label';
                        badge.textContent = page.pageNumber;
                        button.appendChild(badge);
                    }

                    grid.appendChild(button);

                    if (img.dataset.src) {
                        if (priorityIndices.has(listIndex)) {
                            loadThumbnailImage(img, 0);
                        } else if (observer) {
                            observer.observe(img);
                        } else {
                            loadThumbnailImage(img, 1);
                        }
                    }
                });

                if (scrollContainer) {
                    scrollContainer.scrollTop = previousScrollTop;
                }

                lastRenderedBookUuid = normalizedBookUuid;
            } else {
                updateThumbnailLabels(normalizedPages);
            }

            highlightActiveThumbnail(currentPage ? currentPage.uuid : null);
            updatePageNumberInput();
            scheduleThumbnailDrawerHeightSync();
        }

        function handlePageNumberJump(rawValue) {
            const input = document.getElementById('page-number-input');
            if (!input || !bookPages.length) {
                updatePageNumberInput();
                return;
            }

            const value = rawValue !== undefined ? rawValue : input.value;
            const parsed = Number.parseInt(String(value).trim(), 10);

            if (!Number.isFinite(parsed)) {
                updatePageNumberInput();
                return;
            }

            const boundedIndex = Math.min(Math.max(parsed - 1, 0), bookPages.length - 1);
            const target = bookPages[boundedIndex];

            if (!target || !target.uuid) {
                updatePageNumberInput();
                return;
            }

            if (currentPage && currentPage.uuid === target.uuid) {
                updatePageNumberInput();
                return;
            }

            navigateToUuid(target.uuid);
        }

        function setupPageNumberJump() {
            const input = document.getElementById('page-number-input');
            if (!input) {
                return;
            }

            input.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    handlePageNumberJump(input.value);
                }
            });

            input.addEventListener('change', () => {
                handlePageNumberJump(input.value);
            });

            input.addEventListener('blur', () => {
                updatePageNumberInput();
            });

            updatePageNumberInput();
        }

        function initializeThumbnailDrawer() {
            const toggle = document.getElementById('thumbnail-toggle');
            if (!toggle) {
                thumbnailDrawerCollapsed = false;
                return;
            }

            toggle.addEventListener('click', () => {
                setThumbnailDrawerCollapsed(!thumbnailDrawerCollapsed);
            });

            setThumbnailDrawerCollapsed(false);
        }

        function setToolsVisible(show) {
            const tools = document.getElementById("page-tools");
            if (tools) {
                tools.style.display = show ? "flex" : "none";
            }
        }

        function setLoadingState(active) {
            const container = document.querySelector('.container');
            const loading = document.getElementById('loading');
            const isActive = Boolean(active);

            if (container) {
                container.classList.toggle('is-loading', isActive);
            }
            if (loading) {
                loading.setAttribute('aria-hidden', isActive ? 'false' : 'true');
            }
            if (!isActive) {
                scheduleThumbnailDrawerHeightSync();
            }
        }

        function setLargePreviewActive() {
            const container = document.getElementById("page-preview");
            const largeBox = document.getElementById("preview-large");

            if (!container || !largeBox) {
                return;
            }

            const isHovered = container.matches(":hover") || largeBox.matches(":hover") || container.matches(":focus-within");
            const isActive = isHovered && container.classList.contains("preview-loaded");

            if (isActive) {
                largeBox.classList.add("preview-large-visible");
            } else {
                largeBox.classList.remove("preview-large-visible");
            }

            largeBox.setAttribute("aria-hidden", isActive ? "false" : "true");
        }

        function resetPreview() {
            const container = document.getElementById("page-preview");
            const thumb = document.getElementById("preview-image-thumb");
            const largeImg = document.getElementById("preview-image-large");
            const largeBox = document.getElementById("preview-large");

            setLargePreviewActive();

            if (previewObjectUrl) {
                URL.revokeObjectURL(previewObjectUrl);
                previewObjectUrl = null;
            }

            previewImageUuid = null;
            previewFetchToken = null;

            if (thumb) {
                thumb.onload = null;
                thumb.onerror = null;
                thumb.src = "";
                thumb.style.display = "none";
                thumb.style.width = "";
                thumb.style.height = "";
                thumb.style.maxWidth = "";
                thumb.style.maxHeight = "";
                thumb.style.minHeight = "";
                thumb.style.opacity = "";
            }

            if (largeImg) {
                largeImg.onload = null;
                largeImg.src = "";
                largeImg.style.width = "";
                largeImg.style.height = "";
                largeImg.style.maxWidth = "";
                largeImg.style.maxHeight = "";
            }

            if (largeBox) {
                largeBox.style.width = "";
                largeBox.style.maxWidth = "";
                largeBox.style.height = "";
            }

            if (container) {
                container.style.display = "none";
                container.classList.remove("preview-visible", "preview-loaded", "preview-error", "preview-has-status");
                delete container.dataset.previewStream;
            }
            updatePreviewStatus("");
        }

        function updatePreviewStatus(message) {
            const status = document.getElementById('preview-status');
            const container = document.getElementById('page-preview');
            if (!status || !container) {
                return;
            }
            status.textContent = message || "";
            const hasMessage = Boolean(message);
            container.classList.toggle('preview-has-status', hasMessage);
        }

        function computeThumbMaxHeight() {
            const layout = document.querySelector('.page-info-layout');
            if (layout) {
                const rect = layout.getBoundingClientRect();
                if (rect && rect.height > 0) {
                    return rect.height;
                }
            }
            const details = document.querySelector('.page-details');
            if (details) {
                const rect = details.getBoundingClientRect();
                if (rect && rect.height > 0) {
                    return rect.height;
                }
            }
            const section = document.getElementById("page-info");
            if (section) {
                const rect = section.getBoundingClientRect();
                if (rect && rect.height > 0) {
                    return rect.height;
                }
            }
            return 260;
        }

        function computeLargePreviewWidth() {
            const resultBox = document.querySelector('#results .result-box');
            if (resultBox && resultBox.offsetWidth) {
                return Math.round(resultBox.offsetWidth);
            }

            const container = document.querySelector('.container');
            if (container && container.offsetWidth) {
                const containerWidth = container.offsetWidth;
                const fallback = Math.min(containerWidth * 0.5, window.innerWidth * 0.9);
                return Math.round(Math.max(fallback, 360));
            }

            return Math.round(Math.min(window.innerWidth * 0.6, 900));
        }

        function applyLargePreviewSizing(img, box) {
            if (!img || !box) {
                return;
            }

            const maxWidth = computeLargePreviewWidth();
            const naturalWidth = img.naturalWidth || 0;
            const naturalHeight = img.naturalHeight || 0;
            const maxViewportHeight = Math.max(Math.round(window.innerHeight * 0.9), 320);

            box.style.maxWidth = `${Math.round(maxWidth)}px`;

            if (naturalWidth > 0 && naturalHeight > 0) {
                const ratio = naturalHeight / naturalWidth;
                if (ratio > 0) {
                    let targetWidth = maxWidth;
                    let targetHeight = Math.round(targetWidth * ratio);

                    if (targetHeight > maxViewportHeight) {
                        targetHeight = maxViewportHeight;
                        targetWidth = Math.round(targetHeight / ratio);
                    }

                    const safeWidth = Math.max(200, targetWidth);
                    const safeHeight = Math.max(1, targetHeight);

                    box.style.width = `${safeWidth}px`;
                    box.style.height = `${safeHeight}px`;
                    img.style.width = `${safeWidth}px`;
                    img.style.height = `${safeHeight}px`;
                    return;
                }
            }

            const fallbackWidth = Math.max(320, Math.round(Math.min(maxWidth, 720)));
            box.style.width = `${fallbackWidth}px`;
            box.style.height = "auto";
            img.style.width = `${fallbackWidth}px`;
            img.style.height = "auto";
        }

        function refreshLargePreviewSizing() {
            const container = document.getElementById("page-preview");
            const largeImg = document.getElementById("preview-image-large");
            const largeBox = document.getElementById("preview-large");

            if (!container || !largeImg || !largeBox) {
                return;
            }

            if (container.classList.contains("preview-loaded") && largeImg.complete && largeImg.naturalWidth > 0) {
                applyLargePreviewSizing(largeImg, largeBox);
            } else if (!container.classList.contains("preview-loaded")) {
                largeBox.style.width = "";
                largeBox.style.height = "";
                largeImg.style.width = "";
                largeImg.style.height = "";
            }
        }

        function sizeThumbnail(thumb, maxWidth) {
            const computedHeight = computeThumbMaxHeight();
            const safeHeight = Math.min(Math.max(Number.isFinite(computedHeight) ? computedHeight : 0, 120), 180);
            const safeWidth = Math.max(Number.isFinite(maxWidth) ? maxWidth : 0, 220);

            thumb.style.height = "auto";
            thumb.style.maxHeight = `${Math.round(safeHeight)}px`;
            thumb.style.width = "auto";
            thumb.style.maxWidth = `${Math.round(safeWidth)}px`;
            thumb.style.minHeight = "120px";
            thumb.style.opacity = "1";
            thumb.style.display = "block";
        }

        async function fetchPreviewImage(uuid) {
            const streamOrder = ["AUTO", "IMG_PREVIEW", "IMG_THUMB", "IMG_FULL"];
            let lastError = null;

            for (const requestedStream of streamOrder) {
                try {
                    const response = await fetch(`/preview?uuid=${encodeURIComponent(uuid)}&stream=${requestedStream}`, { cache: "no-store" });
                    if (!response.ok) {
                        lastError = new Error(`HTTP ${response.status} (${requestedStream})`);
                        continue;
                    }

                    const contentTypeHeader = response.headers.get("Content-Type") || "";
                    const contentType = contentTypeHeader.toLowerCase();

                    if (!contentType.startsWith("image/")) {
                        lastError = new Error(`Neočekávaný obsah (${requestedStream}): ${contentTypeHeader || 'bez Content-Type'}`);
                        continue;
                    }

                    if (contentType.includes("jp2")) {
                        lastError = new Error(`Formát ${contentTypeHeader} není prohlížečem podporovaný (${requestedStream}).`);
                        continue;
                    }

                    const blob = await response.blob();
                    const actualStream = response.headers.get("X-Preview-Stream") || requestedStream;

                    return {
                        blob,
                        stream: actualStream,
                        contentType: contentTypeHeader,
                    };
                } catch (error) {
                    lastError = error;
                }
            }

            throw lastError || new Error("Náhled se nepodařilo načíst.");
        }

        function showPreviewFromCache(entry) {
            const container = document.getElementById("page-preview");
            const thumb = document.getElementById("preview-image-thumb");
            const largeImg = document.getElementById("preview-image-large");
            const largeBox = document.getElementById("preview-large");

            if (!container || !thumb || !largeImg || !largeBox || !previewObjectUrl) {
                return;
            }

            const cached = entry || previewCache.get(previewImageUuid) || null;
            if (cached && cached.stream) {
                container.dataset.previewStream = cached.stream;
            }

            largeImg.onload = null;
            const handleLargeLoad = () => {
                applyLargePreviewSizing(largeImg, largeBox);
            };
            largeImg.addEventListener("load", handleLargeLoad, { once: true });
            largeImg.src = previewObjectUrl;

            applyLargePreviewSizing(largeImg, largeBox);

            if (largeImg.complete && largeImg.naturalWidth > 0) {
                applyLargePreviewSizing(largeImg, largeBox);
            }

            const finalize = () => {
                thumb.style.opacity = "1";
                thumb.onload = null;
            };

            if (thumb.src !== previewObjectUrl) {
                thumb.style.opacity = "0";
                thumb.onload = finalize;
                thumb.src = previewObjectUrl;
                thumb.onerror = () => {
                    updatePreviewStatus("Náhled se nepodařilo načíst.");
                    container.classList.add("preview-error");
                    container.classList.remove("preview-loaded");
                    setLargePreviewActive(false);
                };
                if (thumb.complete && thumb.naturalWidth > 0) {
                    finalize();
                } else if (thumb.complete) {
                    updatePreviewStatus("Náhled se nepodařilo načíst.");
                    container.classList.add("preview-error");
                    container.classList.remove("preview-loaded");
                    setLargePreviewActive();
                }
            } else if (thumb.naturalWidth > 0) {
                finalize();
            } else {
                thumb.style.opacity = "0";
                thumb.onload = finalize;
                thumb.onerror = () => {
                    updatePreviewStatus("Náhled se nepodařilo načíst.");
                    container.classList.add("preview-error");
                    container.classList.remove("preview-loaded");
                    setLargePreviewActive(false);
                };
            }

            container.style.display = "flex";
            container.classList.add("preview-visible", "preview-loaded");
            container.classList.remove("preview-error");
            updatePreviewStatus("");

            setLargePreviewActive();
            container.style.height = "";
        }

        async function loadPreview(uuid) {
            const container = document.getElementById("page-preview");
            const thumb = document.getElementById("preview-image-thumb");
            const largeImg = document.getElementById("preview-image-large");
            const largeBox = document.getElementById("preview-large");
            if (!container || !thumb || !largeImg || !largeBox || !uuid) {
                return;
            }

            setLargePreviewActive(false);

            const preservedHeight = container.offsetHeight || 0;
            if (preservedHeight > 0) {
                container.style.height = `${preservedHeight}px`;
            }

            if (previewImageUuid === uuid && previewObjectUrl) {
                showPreviewFromCache();
                container.style.height = "";
                return;
            }

            const cachedEntry = previewCache.get(uuid);
            if (cachedEntry && cachedEntry.objectUrl) {
                previewImageUuid = uuid;
                previewObjectUrl = cachedEntry.objectUrl;
                showPreviewFromCache(cachedEntry);
                container.style.height = "";
                return;
            }

            if (previewFetchToken === uuid) {
                return;
            }

            previewFetchToken = uuid;
            previewImageUuid = uuid;

            container.style.display = "flex";
            container.classList.add("preview-visible");
            container.classList.remove("preview-loaded", "preview-error");
            updatePreviewStatus("Načítám náhled...");
            thumb.style.display = "block";
            thumb.style.opacity = "0";

            let handleLoad;

            try {
                const entry = await ensurePreviewEntry(uuid);

                if (!entry || previewImageUuid !== uuid) {
                    container.style.height = "";
                    return;
                }

                const previousUrl = previewObjectUrl;
                previewObjectUrl = entry.objectUrl;

                if (previousUrl && previousUrl !== previewObjectUrl) {
                    const stillReferenced = Array.from(previewCache.values()).some(item => item && item.objectUrl === previousUrl);
                    if (!stillReferenced) {
                        URL.revokeObjectURL(previousUrl);
                    }
                }

                largeImg.onload = null;
                const handleLargeLoad = () => {
                    applyLargePreviewSizing(largeImg, largeBox);
                };
                largeImg.addEventListener("load", handleLargeLoad, { once: true });
                largeImg.src = previewObjectUrl;

                applyLargePreviewSizing(largeImg, largeBox);

                handleLoad = () => {
                    thumb.style.opacity = "1";
                };

                thumb.addEventListener("load", handleLoad, { once: true });
                thumb.onerror = () => {
                    if (previewImageUuid === uuid) {
                        console.error("Chyba při načítání obrázku náhledu");
                        updatePreviewStatus("Náhled se nepodařilo načíst.");
                        container.classList.add("preview-error");
                        container.classList.remove("preview-loaded");
                        setLargePreviewActive();
                    }
                };
                thumb.src = previewObjectUrl;
                thumb.style.opacity = "1";

                if (thumb.complete && thumb.naturalWidth === 0) {
                    if (previewImageUuid === uuid) {
                        updatePreviewStatus("Náhled se nepodařilo načíst.");
                        container.classList.add("preview-error");
                        container.classList.remove("preview-loaded");
                        setLargePreviewActive();
                    }
                }

                container.dataset.previewStream = entry.stream;

                container.classList.add("preview-loaded");
                updatePreviewStatus("");

                setLargePreviewActive();
            } catch (error) {
                if (previewImageUuid === uuid) {
                    console.error("Chyba při načítání náhledu:", error);
                    updatePreviewStatus("Náhled se nepodařilo načíst.");
                    container.classList.add("preview-error");
                    setLargePreviewActive(false);
                }
            } finally {
                if (previewFetchToken === uuid) {
                    previewFetchToken = null;
                }
                if (handleLoad) {
                    thumb.removeEventListener("load", handleLoad);
                }
                container.style.height = "";
            }
        }

        function refreshPagePosition() {
            const position = document.getElementById("page-position");
            if (!position) {
                return;
            }
            if (currentPage && typeof currentPage.index === "number" && navigationState && typeof navigationState.total === "number" && navigationState.total > 0) {
                position.textContent = `${currentPage.index + 1} / ${navigationState.total}`;
            } else if (currentPage && typeof currentPage.index === "number") {
                position.textContent = `${currentPage.index + 1}`;
            } else {
                position.textContent = "-";
            }
        }

        function updateNavigationControls(nav) {
            const prevBtn = document.getElementById("prev-page");
            const nextBtn = document.getElementById("next-page");

            navigationState = nav || null;

            if (!prevBtn || !nextBtn) {
                return;
            }

            if (!navigationState) {
                prevBtn.disabled = true;
                nextBtn.disabled = true;
                refreshPagePosition();
                return;
            }

            prevBtn.disabled = !navigationState.hasPrev;
            nextBtn.disabled = !navigationState.hasNext;
            refreshPagePosition();
        }

        function updateBookInfo() {
            const section = document.getElementById("book-info");
            const titleEl = document.getElementById("book-title");
            const handleEl = document.getElementById("book-handle");
            const libraryEl = document.getElementById("book-library");
            const metadataList = document.getElementById("book-metadata");
            const metadataEmpty = document.getElementById("book-metadata-empty");
            const constantsContainer = document.getElementById("book-constants");

            if (!section || !titleEl || !handleEl || !metadataList || !metadataEmpty) {
                return;
            }

            if (!currentBook) {
                section.style.display = "none";
                handleEl.textContent = "";
                if (libraryEl) {
                    libraryEl.textContent = "";
                    libraryEl.style.display = "none";
                }
                metadataList.innerHTML = "";
                metadataEmpty.style.display = "none";
                if (constantsContainer) {
                    constantsContainer.innerHTML = "";
                    constantsContainer.style.display = "none";
                }
                return;
            }

            section.style.display = "block";
            titleEl.textContent = currentBook.title || "(bez názvu)";

            if (currentBook.handle) {
                handleEl.innerHTML = `<a href="${currentBook.handle}" target="_blank" rel="noopener">Otevřít v Krameriovi</a>`;
            } else {
                handleEl.textContent = "";
            }

            if (libraryEl) {
                if (currentLibrary && currentLibrary.label) {
                    libraryEl.textContent = currentLibrary.label;
                    libraryEl.style.display = "block";
                } else {
                    libraryEl.textContent = "";
                    libraryEl.style.display = "none";
                }
            }

            if (constantsContainer) {
                constantsContainer.innerHTML = "";
                const constants = (currentBook.constants && typeof currentBook.constants === "object") ? currentBook.constants : {};
                const chips = [];

                if (constants && typeof constants === "object") {
                    const basic = (constants.basicTextStyle && typeof constants.basicTextStyle === "object") ? constants.basicTextStyle : null;
                    if (basic) {
                        const valueParts = [];
                        if (typeof basic.fontSize === "number" && Number.isFinite(basic.fontSize)) {
                            const sizeText = basic.fontSize % 1 === 0 ? basic.fontSize.toFixed(0) : basic.fontSize.toFixed(1);
                            valueParts.push(`${sizeText} pt`);
                        }
                        if (basic.fontFamily) {
                            valueParts.push(basic.fontFamily);
                        }
                        const styleFlags = [];
                        if (basic.isBold) {
                            styleFlags.push("tučné");
                        }
                        if (basic.isItalic) {
                            styleFlags.push("kurzíva");
                        }
                        if (!styleFlags.length) {
                            styleFlags.push("regular");
                        }
                        valueParts.push(styleFlags.join(", "));

                        const metaParts = [];
                        if (typeof constants.confidence === "number" && Number.isFinite(constants.confidence)) {
                            metaParts.push(`confidence ${constants.confidence}%`);
                        }
                        if (typeof constants.sampledPages === "number" && constants.sampledPages > 0) {
                            const label = constants.sampledPages === 1 ? "1 strana" : `${constants.sampledPages} stran`;
                            metaParts.push(`vzorek ${label}`);
                        }
                        if (typeof constants.linesSampled === "number" && constants.linesSampled > 0) {
                            const lineLabel = constants.linesSampled === 1 ? "1 řádek" : `${constants.linesSampled} řádků`;
                            metaParts.push(lineLabel);
                        }
                        if (typeof constants.distinctStyles === "number" && constants.distinctStyles > 0) {
                            const styleLabel = constants.distinctStyles === 1 ? "1 styl" : `${constants.distinctStyles} stylů`;
                            metaParts.push(styleLabel);
                        }
                        if (basic.styleId) {
                            metaParts.push(`styl ${basic.styleId}`);
                        }

                        chips.push({
                            label: "Typický text",
                            value: valueParts.filter(Boolean).join(" • ") || "neuvedeno",
                            meta: metaParts.filter(Boolean).join(" • "),
                        });
                    }
                }

                if (chips.length) {
                    chips.forEach(item => {
                        const chipEl = document.createElement("div");
                        chipEl.className = "book-chip";
                        chipEl.innerHTML = `<strong>${item.label}</strong><div class="book-chip-value">${item.value}</div>`;
                        if (item.meta) {
                            const metaEl = document.createElement("div");
                            metaEl.className = "book-chip-meta";
                            metaEl.textContent = item.meta;
                            chipEl.appendChild(metaEl);
                        }
                        constantsContainer.appendChild(chipEl);
                    });
                    constantsContainer.style.display = "grid";
                } else {
                    constantsContainer.style.display = "none";
                }
            }

            metadataList.innerHTML = "";
            const mods = Array.isArray(currentBook.mods) ? currentBook.mods : [];

            if (!mods.length) {
                metadataEmpty.style.display = "block";
                metadataEmpty.textContent = "Metadata se nepodařilo načíst.";
            } else {
                metadataEmpty.style.display = "none";
                mods.forEach(entry => {
                    const dt = document.createElement("dt");
                    dt.textContent = entry.label || "---";
                    const dd = document.createElement("dd");
                    dd.textContent = entry.value || "";
                    metadataList.appendChild(dt);
                    metadataList.appendChild(dd);
                });
            }
        }

        function updatePageInfo() {
            const section = document.getElementById("page-info");
            const summary = document.getElementById("page-summary");
            const side = document.getElementById("page-side");
            const uuidEl = document.getElementById("page-uuid");
            const handleEl = document.getElementById("page-handle");

            if (!section || !summary || !side || !uuidEl || !handleEl) {
                return;
            }

            if (!currentPage) {
                section.style.display = "none";
                summary.textContent = "";
                side.textContent = "";
                uuidEl.textContent = "";
                handleEl.textContent = "";
                resetPreview();
                setToolsVisible(false);
                refreshPagePosition();
                highlightActiveThumbnail(null);
                updatePageNumberInput();
                return;
            }

            section.style.display = "block";

            const parts = [];
            if (currentPage.pageNumber) {
                parts.push(`Strana: ${currentPage.pageNumber}`);
            }
            if (currentPage.pageType) {
                parts.push(`Typ: ${currentPage.pageType}`);
            }
            summary.textContent = parts.length ? parts.join(" • ") : "Informace o straně nejsou k dispozici.";

            if (currentPage.pageSide) {
                side.textContent = `Pozice: ${currentPage.pageSide}`;
            } else {
                side.textContent = "Pozice: neznámá (API neposkytuje údaj).";
            }

            uuidEl.textContent = currentPage.uuid ? `UUID: ${currentPage.uuid}` : "";

            if (currentPage.handle) {
                handleEl.innerHTML = `<a href="${currentPage.handle}" target="_blank" rel="noopener">Otevřít stránku v Krameriovi</a>`;
            } else {
                handleEl.textContent = "";
            }

            const previewContainer = document.getElementById("page-preview");
            if (previewContainer) {
                previewContainer.style.display = "flex";
                previewContainer.classList.add("preview-visible");
            }

            setToolsVisible(true);
            refreshPagePosition();
        }

        function goToAdjacent(direction) {
            if (!navigationState) {
                return;
            }
            const targetUuid = direction === "prev" ? navigationState.prevUuid : navigationState.nextUuid;
            if (!targetUuid) {
                return;
            }
            navigateToUuid(targetUuid);
        }

        function elementConsumesTextInput(element) {
            if (!element) {
                return false;
            }
            if (element === document.body) {
                return false;
            }
            if (element.isContentEditable) {
                return true;
            }
            const tag = element.tagName ? element.tagName.toUpperCase() : "";
            if (!tag) {
                return false;
            }
            if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
                return true;
            }
            return false;
        }

        function setUuidButtonsState() {
            const uuidField = document.getElementById("uuid");
            const copyBtn = document.getElementById("uuid-copy");
            const clearBtn = document.getElementById("uuid-clear");
            const hasValue = Boolean(uuidField && uuidField.value.trim().length);
            if (copyBtn) copyBtn.disabled = !hasValue;
            if (clearBtn) clearBtn.disabled = !hasValue;
        }

        function copyCurrentUuid() {
            const uuidField = document.getElementById("uuid");
            if (!uuidField) {
                return;
            }
            const value = uuidField.value.trim();
            if (!value) {
                showTooltip('uuid-copy-tooltip', 'UUID je prázdné');
                return;
            }
            const fallbackCopy = () => {
                try {
                    const currentSelectionStart = uuidField.selectionStart;
                    const currentSelectionEnd = uuidField.selectionEnd;
                    uuidField.focus();
                    uuidField.select();
                    document.execCommand("copy");
                    if (typeof currentSelectionStart === "number" && typeof currentSelectionEnd === "number") {
                        uuidField.setSelectionRange(currentSelectionStart, currentSelectionEnd);
                    } else {
                        uuidField.setSelectionRange(value.length, value.length);
                    }
                    showTooltip('uuid-copy-tooltip', 'UUID zkopírováno');
                    return true;
                } catch (error) {
                    console.warn("Nelze zkopírovat UUID do schránky:", error);
                    showTooltip('uuid-copy-tooltip', 'Kopírování selhalo', 2600);
                    return false;
                }
            };

            if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
                navigator.clipboard.writeText(value)
                    .then(() => {
                        showTooltip('uuid-copy-tooltip', 'UUID zkopírováno');
                    })
                    .catch(() => fallbackCopy());
            } else {
                fallbackCopy();
            }
        }

        function clearUuidInput() {
            const uuidField = document.getElementById("uuid");
            if (!uuidField) {
                return;
            }
            uuidField.value = "";
            uuidField.focus();
            setUuidButtonsState();
        }

        async function pasteUuidFromClipboard() {
            const uuidField = document.getElementById("uuid");
            if (!uuidField) {
                return;
            }

            const fallback = () => {
                uuidField.focus();
                console.warn('Schránku nelze přečíst - použijte klávesovou zkratku pro vložení.');
                showTooltip('uuid-paste-tooltip', 'Schránku nelze přečíst', 2600);
            };

            if (navigator.clipboard && typeof navigator.clipboard.readText === "function") {
                try {
                    const text = await navigator.clipboard.readText();
                    const trimmed = (text || "").trim();
                    uuidField.value = trimmed;
                    setUuidButtonsState();
                    if (trimmed) {
                        showTooltip('uuid-paste-tooltip', 'UUID vloženo');
                        handleLoadClick();
                    } else {
                        showTooltip('uuid-paste-tooltip', 'Schránka je prázdná', 2000);
                        uuidField.focus();
                    }
                    return;
                } catch (error) {
                    console.warn('Nelze načíst obsah schránky:', error);
                }
            }

            fallback();
        }

        function blurUuidField() {
            const uuidField = document.getElementById("uuid");
            if (uuidField && typeof uuidField.blur === "function") {
                uuidField.blur();
            }
        }

        function isModalActive() {
            const modal = document.getElementById("alto-modal");
            return Boolean(modal && modal.style.display === "block");
        }

        function handleLoadClick() {
            blurUuidField();
            processAlto();
        }

        function setupKeyboardShortcuts() {
            document.addEventListener("keydown", (event) => {
                if (event.defaultPrevented) {
                    return;
                }

                const key = event.key;
                const activeElement = document.activeElement;
                const modalVisible = isModalActive();
                const isWithinModal = modalVisible && activeElement && activeElement.closest("#alto-modal");

                if (event.altKey || event.ctrlKey || event.metaKey) {
                    return;
                }

                if ((key === "ArrowLeft" || key === "ArrowRight") && !isWithinModal) {
                    if (elementConsumesTextInput(activeElement)) {
                        return;
                    }
                    event.preventDefault();
                    goToAdjacent(key === "ArrowLeft" ? "prev" : "next");
                    return;
                }

                if (key === "Enter" && !isWithinModal) {
                    if (activeElement && (activeElement.tagName === "TEXTAREA" || activeElement.isContentEditable)) {
                        return;
                    }
                    const loadButton = document.getElementById("load-button");
                    if (loadButton && !loadButton.disabled) {
                        event.preventDefault();
                        handleLoadClick();
                    } else {
                        event.preventDefault();
                        blurUuidField();
                        processAlto();
                    }
                }
            });
        }

        function applyProcessResult(uuid, data, previousScrollY, toolsElement) {
            cacheProcessData(uuid, data);

            currentAltoXml = data.alto_xml || "";
            const altoBtn = document.getElementById("alto-preview-btn");
            if (altoBtn) {
                altoBtn.style.display = currentAltoXml ? "block" : "none";
            }

            currentBook = data.book || null;
            currentPage = data.currentPage || null;
            currentLibrary = data.library || (currentBook && currentBook.library) || null;

            if (currentBook && currentLibrary && !currentBook.library) {
                currentBook.library = currentLibrary;
            }
            if (currentPage && currentLibrary && !currentPage.library) {
                currentPage.library = currentLibrary;
            }

            updateBookInfo();
            updatePageInfo();
            updateNavigationControls(data.navigation || null);
            ensureThumbnailGrid(Array.isArray(data.pages) ? data.pages : [], currentBook && currentBook.uuid ? currentBook.uuid : null);

            if (currentPage && currentPage.uuid) {
                loadPreview(currentPage.uuid);
            } else {
                resetPreview();
            }

            currentResults = {
                python: data.python || "",
                typescript: data.typescript || "",
                baseKey: buildResultCacheKey(data.python || "", data.typescript || "", currentPage && currentPage.uuid ? currentPage.uuid : uuid),
            };
            diffRequestToken += 1;
            diffCache.clear();
            clearAgentOutput();
            renderComparisonResults();

            const results = document.getElementById("results");
            if (results) {
                results.style.display = "grid";
            }

            updateDiffToggleState();

            const uuidField = document.getElementById("uuid");
            if (uuidField && currentPage && currentPage.uuid) {
                uuidField.value = currentPage.uuid;
                setUuidButtonsState();
            }

            const autoCheckbox = document.getElementById("agent-auto-correct");
            if (autoCheckbox && autoCheckbox.checked) {
                const attemptAutoRun = (attemptsRemaining) => {
                    const runButton = document.getElementById("agent-run");
                    const select = document.getElementById("agent-select");
                    const hasAgent = Boolean(select && select.value);
                    const hasPython = currentResults && typeof currentResults.python === "string" && currentResults.python.trim().length > 0;
                    if ((runButton && runButton.disabled) || !hasAgent || !hasPython) {
                        if (attemptsRemaining > 0 && autoCheckbox.checked) {
                            window.setTimeout(() => attemptAutoRun(attemptsRemaining - 1), 200);
                        } else {
                            autoRunScheduled = false;
                        }
                        return;
                    }
                    runSelectedAgent({ autoTriggered: true }).catch((error) => {
                        autoRunScheduled = false;
                        console.warn('Automatické spuštění agenta se nezdařilo:', error);
                    });
                };

                if (!autoRunScheduled) {
                    autoRunScheduled = true;
                    attemptAutoRun(5);
                }
            } else {
                autoRunScheduled = false;
            }

            updateCacheWindow(currentPage ? currentPage.uuid : uuid, data.navigation || null);
            schedulePrefetch(data.navigation || null);
        }

        function loadStoredDiffMode() {
            try {
                const stored = localStorage.getItem(DIFF_MODE_STORAGE_KEY);
                if (stored === DIFF_MODES.WORD || stored === DIFF_MODES.CHAR) {
                    return stored;
                }
            } catch (error) {
                console.warn('Nelze načíst uložený režim diffu:', error);
            }
            return DIFF_MODES.NONE;
        }

        function persistDiffMode(mode) {
            try {
                if (mode === DIFF_MODES.WORD || mode === DIFF_MODES.CHAR) {
                    localStorage.setItem(DIFF_MODE_STORAGE_KEY, mode);
                } else {
                    localStorage.removeItem(DIFF_MODE_STORAGE_KEY);
                }
            } catch (error) {
                console.warn('Nelze uložit režim diffu:', error);
            }
        }

        function updateDiffToggleState() {
            const container = document.getElementById("diff-mode-controls");
            if (!container) {
                return;
            }
            const buttons = container.querySelectorAll('.diff-toggle');
            buttons.forEach((button) => {
                if (!(button instanceof HTMLElement)) {
                    return;
                }
                const mode = button.getAttribute('data-diff-mode');
                const isActive = mode === diffMode;
                button.classList.toggle('is-active', Boolean(isActive));
                button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
                button.dataset.diffActive = isActive ? 'true' : 'false';
            });
        }

        function setDiffMode(newMode) {
            const normalized = newMode === DIFF_MODES.WORD || newMode === DIFF_MODES.CHAR ? newMode : DIFF_MODES.NONE;
            const hasChanged = diffMode !== normalized;
            diffMode = normalized;
            if (diffMode === DIFF_MODES.NONE) {
                persistDiffMode(null);
            } else {
                persistDiffMode(diffMode);
            }
            diffCache.clear();
            updateDiffToggleState();
            if (hasChanged) {
                renderComparisonResults();
            }
        }

        function initializeDiffControls() {
            diffMode = loadStoredDiffMode();
            const container = document.getElementById("diff-mode-controls");
            if (!container) {
                return;
            }
            container.addEventListener('click', (event) => {
                const target = event.target;
                if (!target || !(target instanceof HTMLElement)) {
                    return;
                }
                if (!target.matches('.diff-toggle')) {
                    return;
                }
                const requestedMode = target.getAttribute('data-diff-mode');
                if (!requestedMode) {
                    return;
                }
                const nextMode = diffMode === requestedMode ? DIFF_MODES.NONE : requestedMode;
                setDiffMode(nextMode);
            });
            updateDiffToggleState();
        }

        function computeSimpleHash(text) {
        if (!text) {
            return 0;
        }
        let hash = 0;
        for (let index = 0; index < text.length; index += 1) {
            hash = ((hash << 5) - hash) + text.charCodeAt(index);
            hash |= 0;
        }
        return hash >>> 0;
    }

        function showTooltip(elementId, message, duration = 2200) {
            if (!elementId) {
                return;
            }
            const el = document.getElementById(elementId);
            if (!el) {
                return;
            }
            if (tooltipTimers.has(elementId)) {
                clearTimeout(tooltipTimers.get(elementId));
                tooltipTimers.delete(elementId);
            }
            const text = message ? String(message) : '';
            if (!text) {
                el.classList.remove('is-visible');
                el.textContent = '';
                return;
            }
            el.textContent = text;
            el.classList.add('is-visible');
            const timerId = window.setTimeout(() => {
                el.classList.remove('is-visible');
                tooltipTimers.delete(elementId);
            }, Math.max(1000, duration || 0));
            tooltipTimers.set(elementId, timerId);
        }

        function buildResultCacheKey(pythonHtml, tsHtml, pageUuid) {
            const leftHash = computeSimpleHash(pythonHtml || "");
            const rightHash = computeSimpleHash(tsHtml || "");
            return `${pageUuid || "standalone"}:${leftHash}:${rightHash}`;
        }

        let diffRequestToken = 0;

        function renderComparisonResults() {
            const pythonRendered = document.getElementById("python-result");
            const tsRendered = document.getElementById("typescript-result");
            if (!pythonRendered || !tsRendered) {
                return;
            }

            const pythonHtml = currentResults.python || "";
            const tsHtml = currentResults.typescript || "";

            pythonRendered.innerHTML = pythonHtml
                ? `<pre>${pythonHtml}</pre>`
                : '<div class="muted">Žádná data.</div>';
            tsRendered.innerHTML = tsHtml
                ? `<pre>${tsHtml}</pre>`
                : '<div class="muted">Žádná data.</div>';

            const baseKey = currentResults.baseKey || buildResultCacheKey(
                pythonHtml,
                tsHtml,
                currentPage && currentPage.uuid ? currentPage.uuid : ""
            );
            updateDiffSection(pythonHtml, tsHtml, baseKey);
        }

        function hideDiffSection() {
            const diffSection = document.getElementById("diff-section");
            const pythonHtmlContainer = document.getElementById("python-html");
            const tsHtmlContainer = document.getElementById("typescript-html");
            if (pythonHtmlContainer) {
                pythonHtmlContainer.innerHTML = "";
            }
            if (tsHtmlContainer) {
                tsHtmlContainer.innerHTML = "";
            }
            if (diffSection) {
                diffSection.classList.remove("is-visible");
            }
        }

        async function updateDiffSection(pythonHtml, tsHtml, baseKey) {
            const diffSection = document.getElementById("diff-section");
            const pythonHtmlContainer = document.getElementById("python-html");
            const tsHtmlContainer = document.getElementById("typescript-html");

            if (!diffSection || !pythonHtmlContainer || !tsHtmlContainer) {
                return;
            }

            if (diffMode === DIFF_MODES.NONE) {
                hideDiffSection();
                return;
            }

            const requestedMode = diffMode;
            const cacheKey = `${requestedMode}:${baseKey}`;
            if (diffCache.has(cacheKey)) {
                const cached = diffCache.get(cacheKey) || {};
                pythonHtmlContainer.innerHTML = cached.python || "";
                tsHtmlContainer.innerHTML = cached.typescript || "";
                diffSection.classList.add("is-visible");
                return;
            }

            const requestId = ++diffRequestToken;
            pythonHtmlContainer.innerHTML = '<div class="diff-loading">Načítám diff…</div>';
            tsHtmlContainer.innerHTML = '<div class="diff-loading">Načítám diff…</div>';
            diffSection.classList.add("is-visible");

            try {
                const response = await fetch('/diff', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        python: pythonHtml,
                        typescript: tsHtml,
                        mode: requestedMode,
                    }),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok || !data || data.ok === false || !data.diff) {
                    const message = data && data.error ? data.error : response.statusText || 'Neznámá chyba';
                    throw new Error(message);
                }
                diffCache.set(cacheKey, data.diff);
                if (requestId !== diffRequestToken || diffMode !== requestedMode) {
                    return;
                }
                pythonHtmlContainer.innerHTML = data.diff.python || "";
                tsHtmlContainer.innerHTML = data.diff.typescript || "";
                diffSection.classList.add("is-visible");
            } catch (error) {
                console.error('Chyba při načítání diffu:', error);
                if (requestId === diffRequestToken) {
                    const message = typeof error?.message === 'string' ? error.message : String(error);
                    pythonHtmlContainer.innerHTML = `<div class="diff-error">Diff se nepodařilo načíst: ${escapeHtml(message)}</div>`;
                    tsHtmlContainer.innerHTML = `<div class="diff-error">Diff se nepodařilo načíst.</div>`;
                    diffSection.classList.add("is-visible");
                }
            }
        }

        function escapeHtml(input) {
            if (!input) {
                return "";
            }
            return input.replace(/[&<>"']/g, (char) => {
                switch (char) {
                    case '&':
                        return '&amp;';
                    case '<':
                        return '&lt;';
                    case '>':
                        return '&gt;';
                    case '"':
                        return '&quot;';
                    case "'":
                        return '&#39;';
                    default:
                        return char;
                }
            });
        }

        async function processAlto() {
            const uuidField = document.getElementById("uuid");
            const uuid = uuidField ? uuidField.value.trim() : "";

            if (!uuid) {
                alert("Zadejte UUID");
                return;
            }

            const token = ++processRequestToken;
            const previousScrollY = window.pageYOffset || window.scrollY || 0;
            const toolsElement = document.getElementById("page-tools");
            const shouldShowLoading = !pageCache.has(uuid);

            if (shouldShowLoading) {
                setLoadingState(true);
            }

            try {
                const data = await ensureProcessData(uuid);

                if (token !== processRequestToken) {
                    return;
                }

                applyProcessResult(uuid, data, previousScrollY, toolsElement);
            } catch (error) {
                if (token !== processRequestToken) {
                    return;
                }
                console.error("Chyba při zpracování:", error);
                const message = error && error.message ? error.message : String(error);
                alert("Chyba při zpracování: " + message);
                window.requestAnimationFrame(() => {
                    window.scrollTo(0, previousScrollY);
                });
            } finally {
                if (token === processRequestToken) {
                    setLoadingState(false);
                }
            }
        }

        function makeDraggable(element, handle) {
            let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
            const dragHandle = handle || element;
            dragHandle.onmousedown = dragMouseDown;
            function dragMouseDown(e) {
                e.preventDefault();
                pos3 = e.clientX;
                pos4 = e.clientY;
                document.onmouseup = closeDragElement;
                document.onmousemove = elementDrag;
            }
            function elementDrag(e) {
                e.preventDefault();
                pos1 = pos3 - e.clientX;
                pos2 = pos4 - e.clientY;
                pos3 = e.clientX;
                pos4 = e.clientY;
                element.style.top = (element.offsetTop - pos2) + "px";
                element.style.left = (element.offsetLeft - pos1) + "px";
            }
            function closeDragElement() {
                document.onmouseup = null;
                document.onmousemove = null;
            }
        }

        window.onload = function () {
            const prev = document.getElementById("prev-page");
            const next = document.getElementById("next-page");
            if (prev) {
                prev.addEventListener("click", () => goToAdjacent("prev"));
            }
            if (next) {
                next.addEventListener("click", () => goToAdjacent("next"));
            }

            const uuidField = document.getElementById("uuid");
            const uuidCopyBtn = document.getElementById("uuid-copy");
            const uuidPasteBtn = document.getElementById("uuid-paste");
            const uuidClearBtn = document.getElementById("uuid-clear");
            if (uuidField) {
                uuidField.addEventListener("input", () => setUuidButtonsState());
            }
            if (uuidCopyBtn) {
                uuidCopyBtn.addEventListener("click", copyCurrentUuid);
            }
            if (uuidPasteBtn) {
                uuidPasteBtn.addEventListener("click", () => {
                    pasteUuidFromClipboard().catch((error) => {
                        console.warn('Vložení UUID ze schránky se nezdařilo:', error);
                    });
                });
            }
            if (uuidClearBtn) {
                uuidClearBtn.addEventListener("click", clearUuidInput);
            }
            setUuidButtonsState();

            setupPageNumberJump();
            initializeThumbnailDrawer();
            initializeDiffControls();
            initializeAgentUI();
            initializeAgentDiffControls();

            const previewContainer = document.getElementById("page-preview");
            const largeBox = document.getElementById("preview-large");
            if (previewContainer) {
                const handleEnter = () => setLargePreviewActive(true);
                const handleLeave = () => setLargePreviewActive(false);

                previewContainer.addEventListener("pointerenter", handleEnter);
                previewContainer.addEventListener("pointerleave", handleLeave);
                previewContainer.addEventListener("mouseenter", handleEnter);
                previewContainer.addEventListener("mouseleave", handleLeave);
                previewContainer.addEventListener("focusin", handleEnter);
                previewContainer.addEventListener("focusout", handleLeave);
            }
            if (largeBox) {
                const handleEnter = () => setLargePreviewActive(true);
                const handleLeave = () => setLargePreviewActive(false);

                largeBox.addEventListener("pointerenter", handleEnter);
                largeBox.addEventListener("pointerleave", handleLeave);
                largeBox.addEventListener("mouseenter", handleEnter);
                largeBox.addEventListener("mouseleave", handleLeave);
                largeBox.addEventListener("focusin", handleEnter);
                largeBox.addEventListener("focusout", handleLeave);
            }

            const altoBtn = document.getElementById("alto-preview-btn");
            if (altoBtn) {
                altoBtn.style.color = "#007bff";
                altoBtn.style.cursor = "pointer";
                altoBtn.style.textDecoration = "underline";
            altoBtn.addEventListener("click", () => {
                const modal = document.getElementById("alto-modal");
                const content = document.getElementById("alto-content");
                if (modal && content) {
                    content.textContent = currentAltoXml;
                    modal.style.display = "block";
                    modal.focus();
                    const modalContent = modal.querySelector('.modal-content');
                    const modalHeader = modal.querySelector('.modal-header');
                    if (modalContent) {
                        // Set initial position to center
                        modalContent.style.top = "";
                        modalContent.style.left = "";
                        modalContent.style.transform = 'translate(-50%, -50%)';
                        // Make draggable only on header
                        if (modalHeader) {
                            makeDraggable(modalContent, modalHeader);
                        }
                    }
                }
            });
            }

            const closeBtn = document.querySelector(".close");
            if (closeBtn) {
                closeBtn.addEventListener("click", () => {
                    const modal = document.getElementById("alto-modal");
                    if (modal) modal.style.display = "none";
                });
            }

            const modal = document.getElementById("alto-modal");
            if (modal) {
                modal.addEventListener("keydown", (e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key === "a") {
                        e.preventDefault();
                        const content = document.getElementById("alto-content");
                        if (content) {
                            const range = document.createRange();
                            range.selectNodeContents(content);
                            const selection = window.getSelection();
                            selection.removeAllRanges();
                            selection.addRange(range);
                        }
                    }
                });
            }

            setupKeyboardShortcuts();

            window.addEventListener("click", (event) => {
                const modal = document.getElementById("alto-modal");
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            });

            processAlto();
        };

        window.addEventListener("resize", () => {
            refreshLargePreviewSizing();
            setLargePreviewActive();
            scheduleThumbnailDrawerHeightSync(true);
        });
    </script>
    <div id="alto-modal" class="modal" tabindex="-1">
        <div class="modal-content">
            <div class="modal-header">
                <h2>ALTO XML Obsah</h2>
                <span class="close">&times;</span>
            </div>
            <pre id="alto-content"></pre>
        </div>
    </div>
</body>
</html>'''
            html = html.replace('__DEFAULT_AGENT_MODEL__', json.dumps(DEFAULT_MODEL))
            html = html.replace('__DEFAULT_AGENT_PROMPT__', json.dumps(DEFAULT_AGENT_PROMPT_TEXT))
            self.wfile.write(html.encode('utf-8'))

        elif self.path.startswith('/process'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()

            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            uuid = query_params.get('uuid', [''])[0]
            api_base_override = query_params.get('api_base', [''])[0] or None

            if not uuid:
                self.wfile.write(json.dumps({'error': 'UUID je povinný'}).encode('utf-8'))
                return

            try:
                # Create processor with optional api_base_override so the server-side
                # calls are made against the same Kramerius instance the UI selected.
                processor = AltoProcessor(api_base_url=api_base_override)
                context = processor.get_book_context(uuid)

                if not context:
                    self.wfile.write(json.dumps({'error': 'Nepodařilo se načíst metadata pro zadané UUID'}).encode('utf-8'))
                    return

                print(f"Book constants for {uuid}: {context.get('book_constants')}")

                page_uuid = context.get('page_uuid')
                if not page_uuid:
                    self.wfile.write(json.dumps({'error': 'Nepodařilo se určit konkrétní stránku pro zadané UUID'}).encode('utf-8'))
                    return

                book_uuid = context.get('book_uuid')
                # Prefer explicit api_base from context, otherwise use processor's base (which
                # may have been initialized from api_base_override)
                active_api_base = context.get('api_base') or processor.api_base_url
                library_info = describe_library(active_api_base)
                handle_base = library_info.get('handle_base') or ''

                alto_xml = processor.get_alto_data(page_uuid)
                if not alto_xml:
                    self.wfile.write(json.dumps({'error': 'Nepodařilo se stáhnout ALTO data'}).encode('utf-8'))
                    return

                pretty_alto = minidom.parseString(alto_xml).toprettyxml(indent="  ")

                python_result = processor.get_formatted_text(alto_xml, page_uuid, DEFAULT_WIDTH, DEFAULT_HEIGHT)
                typescript_result = simulate_typescript_processing(alto_xml, page_uuid, DEFAULT_WIDTH, DEFAULT_HEIGHT)

                pages = context.get('pages', [])
                current_index = context.get('current_index', -1)
                total_pages = len(pages)
                has_prev = current_index > 0
                has_next = current_index >= 0 and current_index < total_pages - 1

                prev_uuid = pages[current_index - 1]['uuid'] if has_prev else None
                next_uuid = pages[current_index + 1]['uuid'] if has_next else None

                book_data = context.get('book') or {}
                mods_metadata = context.get('mods') or []

                def clean(value: str) -> str:
                    if not value:
                        return ''
                    return ' '.join(value.replace('\xa0', ' ').split())

                page_summary = context.get('page') or {}
                page_item = context.get('page_item') or {}
                book_handle = f"{handle_base}/handle/uuid:{book_uuid}" if handle_base and book_uuid else ''
                page_handle = f"{handle_base}/handle/uuid:{page_uuid}" if handle_base and page_uuid else ''

                page_info = {
                    'uuid': page_uuid,
                    'title': clean(page_summary.get('title') or page_item.get('title') or ''),
                    'pageNumber': clean(page_summary.get('pageNumber') or (page_item.get('details') or {}).get('pagenumber') or ''),
                    'pageType': clean(page_summary.get('pageType') or (page_item.get('details') or {}).get('type') or ''),
                    'pageSide': clean(page_summary.get('pageSide') or (page_item.get('details') or {}).get('pageposition') or (page_item.get('details') or {}).get('pagePosition') or (page_item.get('details') or {}).get('pagerole') or ''),
                    'index': current_index,
                    'iiif': page_item.get('iiif'),
                    'handle': page_handle,
                    'library': library_info,
                }

                book_info = {
                    'uuid': context.get('book_uuid'),
                    'title': clean(book_data.get('title') or ''),
                    'model': book_data.get('model'),
                    'handle': book_handle,
                    'mods': mods_metadata,
                    'constants': context.get('book_constants') or {},
                    'library': library_info,
                }

                navigation = {
                    'hasPrev': has_prev,
                    'hasNext': has_next,
                    'prevUuid': prev_uuid,
                    'nextUuid': next_uuid,
                    'total': total_pages,
                }

                response_data = {
                    'python': python_result,
                    'typescript': typescript_result,
                    'book': book_info,
                    'pages': pages,
                    'currentPage': page_info,
                    'navigation': navigation,
                    'alto_xml': pretty_alto,
                    'library': library_info,
                }

                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))

            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

        elif self.path.startswith('/preview'):
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            uuid = query_params.get('uuid', [''])[0]
            stream = query_params.get('stream', ['IMG_PREVIEW'])[0]
            allowed_streams = {'IMG_THUMB', 'IMG_PREVIEW', 'IMG_FULL', 'AUTO'}

            if not uuid:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'UUID je povinný'}).encode('utf-8'))
                return

            if stream not in allowed_streams:
                stream = 'IMG_PREVIEW'

            candidate_streams = [stream]
            if stream == 'AUTO':
                candidate_streams = ['IMG_FULL', 'IMG_PREVIEW', 'IMG_THUMB']

            candidate_bases = list(dict.fromkeys(DEFAULT_API_BASES))

            last_error = None

            try:
                for candidate in candidate_streams:
                    for base in candidate_bases:
                        upstream_url = f"{base}/item/uuid:{uuid}/streams/{candidate}"
                        response = requests.get(upstream_url, timeout=20)

                        if response.status_code != 200 or not response.content:
                            last_error = f'Nepodařilo se načíst náhled (status {response.status_code} pro {candidate} z {base})'
                            response.close()
                            continue

                        content_type = response.headers.get('Content-Type', 'image/jpeg')
                        if 'jp2' in content_type.lower():
                            last_error = f'Stream {candidate} vrací nepodporovaný formát {content_type}'
                            response.close()
                            continue

                        self.send_response(200)
                        self.send_header('Content-type', content_type)
                        self.send_header('Content-Length', str(len(response.content)))
                        self.send_header('Cache-Control', 'no-store')
                        self.send_header('X-Preview-Stream', candidate)
                        self.end_headers()
                        self.wfile.write(response.content)
                        response.close()
                        return

                self.send_response(502)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                message = last_error or 'Nepodařilo se načíst náhled.'
                self.wfile.write(json.dumps({'error': message}).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(content_length) if content_length else b''
        try:
            payload = json.loads(body.decode('utf-8')) if body else {}
        except Exception:
            payload = {}

        if path == '/diff':
            python_html = ''
            ts_html = ''
            mode = DIFF_MODE_WORD
            if isinstance(payload, dict):
                python_html = str(payload.get('python') or payload.get('python_html') or '')
                ts_html = str(payload.get('typescript') or payload.get('typescript_html') or '')
                requested_mode = payload.get('mode')
                if isinstance(requested_mode, str):
                    mode = requested_mode
            try:
                diff_result = build_html_diff(python_html, ts_html, mode)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {
                    'ok': True,
                    'diff': diff_result,
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            except Exception as err:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(err)}).encode('utf-8'))
            return

        if path == '/agents/diff':
            original_html = ''
            corrected_html = ''
            mode = DIFF_MODE_WORD
            if isinstance(payload, dict):
                original_html = str(payload.get('original') or payload.get('original_html') or '')
                corrected_html = str(payload.get('corrected') or payload.get('corrected_html') or '')
                requested_mode = payload.get('mode')
                if isinstance(requested_mode, str):
                    mode = requested_mode
            try:
                diff_result = build_agent_diff(original_html, corrected_html, mode)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {
                    'ok': True,
                    'diff': diff_result,
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            except Exception as err:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(err)}).encode('utf-8'))
            return

        if path == '/agents/save':
            stored = write_agent_file(payload if isinstance(payload, dict) else {})
            # write_agent_file now returns canonical name on success, or None on failure
            if stored:
                # return the saved agent data back to client for immediate UI sync
                data = read_agent_file(stored) or {}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True, 'stored_name': stored, 'agent': data}, ensure_ascii=False).encode('utf-8'))
                return
            else:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': 'invalid'}).encode('utf-8'))
                return

        if path == '/agents/delete':
            name = payload.get('name') if isinstance(payload, dict) else None
            ok = delete_agent_file(name or '')
            if ok:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': True}).encode('utf-8'))
                return
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': 'not_found'}).encode('utf-8'))
                return
            return

        if path == '/agents/run':
            request_payload = payload if isinstance(payload, dict) else {}
            agent_name = request_payload.get('name')
            agent = read_agent_file(agent_name or '')
            if not agent:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': 'agent_not_found'}).encode('utf-8'))
                return
            model_override = str(request_payload.get('model_override') or '').strip()
            if not model_override:
                model_override = str(request_payload.get('model') or '').strip()
            reasoning_override = str(request_payload.get('reasoning_effort') or '').strip().lower()
            snapshot = request_payload.get('agent_snapshot')
            if isinstance(snapshot, dict):
                agent_for_run = dict(agent)
                agent_for_run.update(snapshot)
            else:
                agent_for_run = dict(agent)
            agent_for_run.setdefault('name', agent_name or '')
            if model_override:
                agent_for_run['model'] = model_override
            if reasoning_override in {'low', 'medium', 'high'}:
                agent_for_run['reasoning_effort'] = reasoning_override
            try:
                result = run_agent_via_responses(agent_for_run, request_payload)
            except AgentRunnerError as err:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(err)}).encode('utf-8'))
                return
            except Exception as err:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'ok': False, 'error': str(err)}).encode('utf-8'))
                return

            response_body = {
                'ok': True,
                'result': result,
                'auto_correct': bool(request_payload.get('auto_correct')),
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response_body, ensure_ascii=False).encode('utf-8'))
            return

def ensure_typescript_build() -> bool:
    if TS_DIST_PATH.exists():
        return True

    npx_path = shutil.which('npx')
    if not npx_path:
        return False

    result = subprocess.run(
        [npx_path, 'tsc'],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        error_output = result.stderr.strip() or result.stdout.strip()
        print(f"TypeScript build failed: {error_output}")
        return False

    return TS_DIST_PATH.exists()


def simulate_typescript_processing(alto_xml: str, uuid: str, width: int, height: int) -> str:
    """Spuštění původní TypeScript logiky přes Node.js"""
    if not ensure_typescript_build():
        return "TypeScript build není k dispozici (zkontrolujte instalaci Node.js a spusťte 'npx tsc')."

    node_path = shutil.which('node')
    if not node_path:
        return "Node.js není dostupný v PATH."

    try:
        completed = subprocess.run(
            [
                node_path,
                str(TS_DIST_PATH),
                'formatted',
                '--stdin',
                '--uuid',
                uuid,
                '--width',
                str(width),
                '--height',
                str(height)
            ],
            input=alto_xml,
            text=True,
            capture_output=True,
            timeout=45,
            cwd=str(ROOT_DIR)
        )

        if completed.returncode != 0:
            error_output = completed.stderr.strip() or completed.stdout.strip()
            return f"TypeScript chyba: {error_output}"

        return completed.stdout.strip()

    except subprocess.TimeoutExpired:
        return "TypeScript zpracování vypršelo (timeout)."
    except Exception as err:
        return f"TypeScript výjimka: {err}"

def run_server(port=8000):
    """Spuštění webového serveru"""
    with ThreadingHTTPServer(("", port), ComparisonHandler) as httpd:
        print(f"Server běží na http://localhost:{port}")
        print("Otevírám prohlížeč...")
        webbrowser.open(f'http://localhost:{port}')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer zastaven")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()
