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
from urllib.parse import urlparse, parse_qs
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import html
import re
from pathlib import Path
from typing import Dict, Optional

# Import původního procesoru
from main_processor import AltoProcessor, DEFAULT_API_BASES
from agent_runner import AgentRunnerError, run_agent as run_agent_via_responses


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
    # parse and clamp numeric fields carefully so that 0 values are preserved
    try:
        temp_val = data.get('temperature') if isinstance(data, dict) else None
        if temp_val is None:
            temperature = 0.0
        else:
            temperature = float(temp_val)
    except Exception:
        temperature = 0.0
    try:
        top_val = data.get('top_p') if isinstance(data, dict) else None
        if top_val is None:
            top_p = 1.0
        else:
            top_p = float(top_val)
    except Exception:
        top_p = 1.0
    # clamp to valid range
    temperature = max(0.0, min(1.0, temperature))
    top_p = max(0.0, min(1.0, top_p))

    safe = {
        'name': nm,
        'display_name': display_name,
        'prompt': str(data.get('prompt') or '')[:200000],
        'temperature': temperature,
        'top_p': top_p,
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
        }
        button:not(.page-thumbnail):not(.thumbnail-toggle):not(.diff-toggle) {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:not(.page-thumbnail):not(.thumbnail-toggle):not(.diff-toggle):hover {
            background-color: #0056b3;
        }
        button:not(.page-thumbnail):not(.thumbnail-toggle):not(.diff-toggle):disabled {
            background-color: #9aa0a6;
            cursor: not-allowed;
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
            border-radius: 4px;
            font-family: "Menlo", "Monaco", "Consolas", "Courier New", monospace;
            padding: 12px;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-x: auto;
            min-height: 64px;
        }
        .diff-html {
            margin: 0;
            line-height: 1.5;
        }
        .diff-added {
            background: rgba(46, 160, 67, 0.25);
            border-radius: 3px;
            padding: 0 2px;
        }
        .diff-removed {
            background: rgba(219, 68, 55, 0.25);
            border-radius: 3px;
            padding: 0 2px;
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

                <div class="form-group">
                    <label for="uuid">UUID stránky nebo dokumentu:</label>
                    <input type="text" id="uuid" placeholder="Zadejte UUID (např. 673320dd-0071-4a03-bf82-243ee206bc0b)" value="673320dd-0071-4a03-bf82-243ee206bc0b">
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
                        <div style="display:flex;gap:12px;align-items:center;">
                            <div style="flex:1;">
                                <label for="agent-temperature">Temperature</label>
                                <div style="display:flex;gap:8px;align-items:center;">
                                    <input id="agent-temperature" type="range" min="0" max="1" step="0.01" value="0.2" style="flex:1;">
                                    <input id="agent-temperature-number" type="number" min="0" max="1" step="0.01" value="0.2" style="width:80px;">
                                </div>
                            </div>
                            <div style="flex:1;">
                                <label for="agent-top-p">Top P</label>
                                <div style="display:flex;gap:8px;align-items:center;">
                                    <input id="agent-top-p" type="range" min="0" max="1" step="0.01" value="1.0" style="flex:1;">
                                    <input id="agent-top-p-number" type="number" min="0" max="1" step="0.01" value="1.0" style="width:80px;">
                                </div>
                            </div>
                        </div>
                        <div style="display:flex;gap:8px;margin-top:12px;">
                            <button id="agent-save" type="button">Uložit agenta</button>
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
        let currentResults = {
            python: "",
            typescript: "",
            baseKey: "",
        };
        const NOTE_STYLE_ATTR = ' style="display:block;font-size:0.82em;color:#1e5aa8;font-weight:bold;"';
        // --- LLM agent management ---
        const AGENTS_STORAGE_KEY = 'altoAgents_v1';
        const AGENT_SELECTED_KEY = 'altoAgentSelected_v1';
        const AGENT_AUTO_CORRECT_KEY = 'altoAgentAutoCorrect_v1';

        // client-side in-memory cache of agents (name -> agent object)
        const agentsCache = {};
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

        async function loadAgents() {
            // if cache already populated, return a shallow map
            const keys = Object.keys(agentsCache);
            if (keys.length) {
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
                (data.agents || []).forEach(a => {
                    // prefer full agent object if present
                    if (a.agent && typeof a.agent === 'object') {
                        agentsCache[a.name] = a.agent;
                    } else {
                        agentsCache[a.name] = agentsCache[a.name] || { name: a.name, display_name: a.display_name || a.name, prompt: '', temperature: 0.2, top_p: 1.0 };
                    }
                    out[a.name] = { name: a.name, display_name: agentsCache[a.name].display_name || a.name };
                });
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

        async function renderAgentSelector() {
            const select = document.getElementById('agent-select');
            const spinnerWrapper = document.getElementById('agent-select-spinner');
            if (spinnerWrapper) spinnerWrapper.style.display = 'inline-block';
            if (!select) {
                if (spinnerWrapper) spinnerWrapper.style.display = 'none';
                return;
            }
            select.innerHTML = '';
            const agents = await loadAgents();
            const selected = loadSelectedAgent();
            const names = Object.keys(agents).sort();
            // If no agents on server, create some defaults client-side and save
            if (!names.length) {
                const defaults = ['editor-default','cleanup-a','semantic-fix'];
                for (const n of defaults) {
                    await fetch('/agents/save', {
                        method: 'POST', headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({ name: n, prompt: DEFAULT_AGENT_PROMPT, temperature: 0.2, top_p: 1.0 })
                    });
                }
                if (spinnerWrapper) spinnerWrapper.style.display = 'none';
                return renderAgentSelector();
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
            try {
                    // prefer in-memory cache where possible
                    if (agentsCache[name]) {
                        setAgentFields(agentsCache[name]);
                        persistSelectedAgent(name);
                        return;
                    }
                    const res = await fetch(`/agents/get?name=${encodeURIComponent(name)}`, { cache: 'no-store' });
                    if (!res.ok) {
                        setAgentFields({name:'', prompt:'', temperature:0.2, top_p:1.0});
                        persistSelectedAgent('');
                        return;
                    }
                    const agent = await res.json();
                    // populate cache and use it
                    agentsCache[name] = agent || {};
                    setAgentFields(agentsCache[name]);
                    persistSelectedAgent(name);
            } catch (err) {
                console.warn('Nelze načíst agenta:', err);
                setAgentFields({name:'', prompt:'', temperature:0.2, top_p:1.0});
            }
        }

        function setAgentFields(agent) {
            const nameEl = document.getElementById('agent-name');
            const promptEl = document.getElementById('agent-prompt');
            const tempRange = document.getElementById('agent-temperature');
            const tempNum = document.getElementById('agent-temperature-number');
            const topRange = document.getElementById('agent-top-p');
            const topNum = document.getElementById('agent-top-p-number');
            if (nameEl) nameEl.value = agent.name || '';
            if (promptEl) {
                const promptValue = (agent.prompt && typeof agent.prompt === 'string' && agent.prompt.trim()) ? agent.prompt : DEFAULT_AGENT_PROMPT;
                promptEl.value = promptValue;
            }
            if (tempRange) tempRange.value = (agent.temperature !== undefined ? agent.temperature : 0.2);
            if (tempNum) tempNum.value = (agent.temperature !== undefined ? agent.temperature : 0.2);
            if (topRange) topRange.value = (agent.top_p !== undefined ? agent.top_p : 1.0);
            if (topNum) topNum.value = (agent.top_p !== undefined ? agent.top_p : 1.0);
        }

        function syncRangeAndNumber(rangeEl, numberEl) {
            if (!rangeEl || !numberEl) return;
            rangeEl.addEventListener('input', () => { numberEl.value = rangeEl.value; });
            numberEl.addEventListener('change', () => {
                let v = parseFloat(numberEl.value);
                if (!Number.isFinite(v)) v = parseFloat(rangeEl.value) || 0;
                v = Math.max(0, Math.min(1, v));
                rangeEl.value = v;
                numberEl.value = v;
            });
        }

        async function saveCurrentAgent() {
            const nameEl = document.getElementById('agent-name');
            const promptEl = document.getElementById('agent-prompt');
            const tempNum = document.getElementById('agent-temperature-number');
            const topNum = document.getElementById('agent-top-p-number');
            if (!nameEl) return;
            const name = String(nameEl.value || '').trim();
            if (!name) {
                alert('Agent must have a name');
                return;
            }
            // parse numeric fields carefully so that 0 is preserved
            let tempVal = parseFloat(tempNum ? tempNum.value : '0.2');
            if (!Number.isFinite(tempVal)) tempVal = 0.0;
            let topVal = parseFloat(topNum ? topNum.value : '1.0');
            if (!Number.isFinite(topVal)) topVal = 1.0;
            const payload = {
                name,
                prompt: promptEl ? promptEl.value : '',
                temperature: tempVal,
                top_p: topVal,
            };
            try {
                const res = await fetch('/agents/save', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
                if (!res.ok) {
                    const txt = await res.text().catch(()=>'');
                    throw new Error('save failed ' + txt);
                }
                const data = await res.json().catch(() => null);
                const stored = (data && data.stored_name) ? data.stored_name : name;
                const agent = (data && data.agent) ? data.agent : payload;

                // update in-memory cache and DOM without re-fetching everything
                agentsCache[stored] = agent;
                if (!agent.display_name) agent.display_name = agent.name || stored;

                const select = document.getElementById('agent-select');
                if (select) {
                    // find existing option with this value
                    let opt = select.querySelector(`option[value="${stored}"]`);
                    if (!opt) {
                        // try to remove old option with previous name if user changed it
                        const oldOpt = select.querySelector(`option[value="${name}"]`);
                        if (oldOpt) oldOpt.remove();
                        opt = document.createElement('option');
                        opt.value = stored;
                        opt.textContent = agent.display_name || stored;
                        select.appendChild(opt);
                    } else {
                        opt.textContent = agent.display_name || stored;
                    }
                    select.value = stored;
                    persistSelectedAgent(stored);
                }

                // update UI fields from saved agent
                setAgentFields(agent || payload);
            } catch (err) {
                alert('Nelze uložit agenta: ' + err);
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
                const currentSel = loadSelectedAgent();
                // remove from in-memory cache and DOM
                delete agentsCache[name];
                const select = document.getElementById('agent-select');
                if (select) {
                    const opt = select.querySelector(`option[value="${name}"]`);
                    if (opt) opt.remove();
                }
                if (currentSel === name) {
                    persistSelectedAgent('');
                    // select first available if any
                    if (select && select.options && select.options.length) {
                        select.selectedIndex = 0;
                        const newSel = select.value;
                        persistSelectedAgent(newSel);
                        if (agentsCache[newSel]) setAgentFields(agentsCache[newSel]);
                    } else {
                        setAgentFields({name:'', prompt:'', temperature:0.2, top_p:1.0});
                    }
                }
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
                switch (type) {
                    case 'h1':
                    case 'h2':
                    case 'h3':
                        tag = type;
                        break;
                    case 'small':
                        tag = 'small';
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
                parts.push(`<${tag}${attrs}>${escapeHtml(text)}</${tag}>`);
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
            container.style.display = 'block';
            statusEl.textContent = statusText || '';
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
            textEl.textContent = bodyText || '';
        }

        function formatHtmlForPre(html) {
            if (!html) {
                return '';
            }
            return escapeHtml(html || '');
        }

        function setAgentResultPanels(originalHtml, correctedContent, correctedIsHtml) {
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
        }

        async function runSelectedAgent() {
            const select = document.getElementById('agent-select');
            if (!select) return;
            const name = select.value;
            if (!name) {
                alert('Žádný agent není vybrán');
                return;
            }
            const pythonHtml = currentResults && currentResults.python ? String(currentResults.python) : '';
            if (!pythonHtml.trim()) {
                alert('Python výstup je prázdný – nejprve načtěte stránku.');
                return;
            }
            const auto = document.getElementById('agent-auto-correct');
            const willAuto = auto && auto.checked;
            persistAutoCorrect(Boolean(willAuto));
            const originalPythonHtml = pythonHtml;

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
            };

            const runBtn = document.getElementById('agent-run');
            const originalLabel = runBtn ? runBtn.textContent : '';

            try {
                if (runBtn) {
                    runBtn.disabled = true;
                    runBtn.textContent = 'Spouštím...';
                }
                setAgentOutput(`Spouštím agenta ${name}...`, '', 'pending');
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
                const usage = result.usage || {};
                const statusParts = [`Agent ${name}`];
                if (result.model) {
                    statusParts.push(`Model: ${result.model}`);
                }
                if (result.stop_reason) {
                    statusParts.push(`Stop reason: ${result.stop_reason}`);
                }
                if (usage.input_tokens !== undefined && usage.output_tokens !== undefined) {
                    statusParts.push(`Tokeny in/out: ${usage.input_tokens}/${usage.output_tokens}`);
                } else if (usage.total_tokens !== undefined) {
                    statusParts.push(`Tokeny celkem: ${usage.total_tokens}`);
                }

                const parsedDoc = parseAgentResultDocument(text);
                const htmlFromDoc = parsedDoc ? documentBlocksToHtml(parsedDoc) : null;
                if (parsedDoc && Array.isArray(parsedDoc.blocks)) {
                    statusParts.push(`Bloků: ${parsedDoc.blocks.length}`);
                }

                const autoRequested = Boolean(data.auto_correct);
                if (autoRequested) {
                    if (htmlFromDoc) {
                        currentResults.python = htmlFromDoc;
                        currentResults.baseKey = buildResultCacheKey(
                            htmlFromDoc,
                            currentResults.typescript || '',
                            currentPage && currentPage.uuid ? currentPage.uuid : ''
                        );
                        renderComparisonResults();
                        statusParts.push('Výsledek aplikován');
                    } else {
                        statusParts.push('Výsledek nelze aplikovat (očekáván JSON se strukturou blocks)');
                    }
                }

                const correctedContent = htmlFromDoc || (text || '');
                const correctedIsHtml = Boolean(htmlFromDoc);
                setAgentResultPanels(originalPythonHtml, correctedContent, correctedIsHtml);

                const statusText = statusParts.join(' · ');
                const displayText = parsedDoc ? JSON.stringify(parsedDoc, null, 2) : (text || '(Agent nevrátil text)');
                setAgentOutput(statusText, displayText, 'success');
            } catch (err) {
                const message = err && err.message ? err.message : String(err);
                setAgentOutput(`Chyba agenta: ${message}`, '', 'error');
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

            if (select) select.addEventListener('change', updateAgentUIFromSelection);
            if (runBtn) runBtn.addEventListener('click', runSelectedAgent);
            if (auto) {
                auto.checked = loadAutoCorrect();
                auto.addEventListener('change', () => persistAutoCorrect(auto.checked));
            }
            if (toggle && settings) {
                toggle.addEventListener('click', async () => {
                    const visible = settings.style.display !== 'none';
                    const nowVisible = !visible;
                    settings.style.display = visible ? 'none' : 'block';
                    toggle.setAttribute('aria-expanded', (nowVisible).toString());
                    // when opening, refresh fields for the currently selected agent
                    if (nowVisible) {
                        try { await updateAgentUIFromSelection(); } catch (e) { /* ignore */ }
                    }
                    // after layout changes, sync thumbnail drawer sizing to account for new page height
                    setTimeout(() => {
                        try { syncThumbnailDrawerHeight(); } catch (e) {}
                    }, 60);
                });
            }
            if (saveBtn) saveBtn.addEventListener('click', async () => { await saveCurrentAgent(); syncThumbnailDrawerHeight(); });
            if (deleteBtn) deleteBtn.addEventListener('click', async () => { await deleteCurrentAgent(); syncThumbnailDrawerHeight(); });

            // wire up syncs
            syncRangeAndNumber(document.getElementById('agent-temperature'), document.getElementById('agent-temperature-number'));
            syncRangeAndNumber(document.getElementById('agent-top-p'), document.getElementById('agent-top-p-number'));

            // Load selector asynchronously; don't block UI interactivity
            renderAgentSelector().catch(() => {});
        }
        function clearThumbnailQueue() {
            thumbnailQueue.length = 0;
        }

        function drainThumbnailQueue() {
            while (thumbnailQueue.length && activeThumbnailLoads < MAX_THUMBNAIL_REQUESTS) {
                const img = thumbnailQueue.shift();
                if (!img || !img.dataset) {
                    continue;
                }
                delete img.dataset.queued;
                startThumbnailLoad(img);
            }
        }

        function finalizeThumbnail(img, success) {
            if (img && img.dataset) {
                delete img.dataset.loading;
                delete img.dataset.queued;
                if (success) {
                    img.dataset.loaded = 'true';
                } else {
                    delete img.dataset.loaded;
                }
            }
            if (activeThumbnailLoads > 0) {
                activeThumbnailLoads -= 1;
            }
            drainThumbnailQueue();
        }

        function startThumbnailLoad(img) {
            if (!img || !img.dataset) {
                return;
            }
            if (img.dataset.loaded === 'true' || img.dataset.loading === 'true') {
                return;
            }
            const src = img.dataset.src;
            if (!src) {
                return;
            }
            img.dataset.loading = 'true';
            activeThumbnailLoads += 1;
            img.src = src;
            if (img.complete) {
                if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                    img.dispatchEvent(new Event('load'));
                } else if (img.naturalWidth === 0) {
                    img.dispatchEvent(new Event('error'));
                }
            }
        }

        function flushAllThumbnailLoads() {
            activeThumbnailLoads = 0;
            clearThumbnailQueue();
        }
        const MAX_THUMBNAIL_REQUESTS = 6;
        const thumbnailQueue = [];
        let activeThumbnailLoads = 0;

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
                syncThumbnailDrawerHeight();
            }
        }

        function navigateToUuid(targetUuid) {
            if (!targetUuid) {
                return;
            }
            const uuidField = document.getElementById("uuid");
            if (uuidField) {
                uuidField.value = targetUuid;
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
            flushAllThumbnailLoads();
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
                        loadThumbnailImage(img);
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

        function loadThumbnailImage(img, immediate = false) {
            if (!img || !img.dataset || img.dataset.loaded === 'true' || !img.dataset.src) {
                return;
            }
            if (img.dataset.loading === 'true') {
                return;
            }
            if (immediate) {
                startThumbnailLoad(img);
            } else if (thumbnailObserver) {
                if (!img.dataset.queued && activeThumbnailLoads >= MAX_THUMBNAIL_REQUESTS) {
                    img.dataset.queued = 'true';
                    thumbnailQueue.push(img);
                } else {
                    startThumbnailLoad(img);
                }
            } else {
                if (activeThumbnailLoads < MAX_THUMBNAIL_REQUESTS) {
                    startThumbnailLoad(img);
                } else if (!img.dataset.queued) {
                    img.dataset.queued = 'true';
                    thumbnailQueue.push(img);
                }
            }
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
                            loadThumbnailImage(img, true);
                        } else if (observer) {
                            observer.observe(img);
                        } else {
                            loadThumbnailImage(img);
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
            syncThumbnailDrawerHeight();
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
                syncThumbnailDrawerHeight();
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

        function buildResultCacheKey(pythonHtml, tsHtml, pageUuid) {
            const leftHash = computeSimpleHash(pythonHtml || "");
            const rightHash = computeSimpleHash(tsHtml || "");
            return `${pageUuid || "standalone"}:${leftHash}:${rightHash}`;
        }

        function renderComparisonResults() {
            const pythonRendered = document.getElementById("python-result");
            const tsRendered = document.getElementById("typescript-result");
            const pythonHtmlContainer = document.getElementById("python-html");
            const tsHtmlContainer = document.getElementById("typescript-html");
            const htmlDiffSection = document.getElementById("html-diff");
            const diffSection = document.getElementById("diff-section");
            if (!pythonRendered || !tsRendered) {
                return;
            }

            const pythonHtml = currentResults.python || "";
            const tsHtml = currentResults.typescript || "";

            // Always render the simple preformatted outputs into the visible result boxes.
            pythonRendered.innerHTML = `<pre>${pythonHtml}</pre>`;
            tsRendered.innerHTML = `<pre>${tsHtml}</pre>`;

            // Only attempt full HTML diff rendering when the HTML containers are present
            // (they were intentionally commented out earlier while preparing the LLM UI).
            if (!pythonHtmlContainer || !tsHtmlContainer) {
                return;
            }

            const baseKey = currentResults.baseKey || buildResultCacheKey(pythonHtml, tsHtml, currentPage && currentPage.uuid ? currentPage.uuid : "");
            const cacheKey = `${diffMode}:${baseKey}`;
            try {
                let cached = diffCache.get(cacheKey);
                if (!cached) {
                    cached = buildDiffMarkup(pythonHtml, tsHtml, diffMode);
                    diffCache.set(cacheKey, cached);
                }

                pythonHtmlContainer.innerHTML = cached.python;
                tsHtmlContainer.innerHTML = cached.typescript;
                if (htmlDiffSection) {
                    htmlDiffSection.style.display = "grid";
                }
                if (diffSection) {
                    diffSection.classList.add('is-visible');
                }
            } catch (error) {
                console.error('Chyba při vykreslování diffu:', error);
                diffMode = DIFF_MODES.NONE;
                persistDiffMode(null);
                diffCache.clear();
                updateDiffToggleState();
                pythonHtmlContainer.innerHTML = wrapCodeContent(escapeHtml(pythonHtml));
                tsHtmlContainer.innerHTML = wrapCodeContent(escapeHtml(tsHtml));
                if (htmlDiffSection) {
                    htmlDiffSection.style.display = "grid";
                }
                if (diffSection) {
                    diffSection.classList.add('is-visible');
                }
            }
        }

        function wrapCodeContent(content, mode) {
            const modeAttr = mode && mode !== DIFF_MODES.NONE ? ` data-diff-mode="${mode}"` : "";
            return `<pre class="diff-content diff-html"${modeAttr}>${content}</pre>`;
        }

        function buildDiffMarkup(pythonHtml, tsHtml, mode) {
            const safePython = pythonHtml || "";
            const safeTs = tsHtml || "";
            if (!mode || mode === DIFF_MODES.NONE) {
                return {
                    python: wrapCodeContent(escapeHtml(safePython), mode),
                    typescript: wrapCodeContent(escapeHtml(safeTs), mode),
                };
            }

            const pythonTokens = tokenizeHtml(safePython);
            const tsTokens = tokenizeHtml(safeTs);
            const operations = diffUsingLcs(pythonTokens, tsTokens, tokensEqual);

            if (mode === DIFF_MODES.WORD) {
                const pythonHighlights = new Array(pythonTokens.length).fill(null);
                const tsHighlights = new Array(tsTokens.length).fill(null);
                operations.forEach((operation) => {
                    if (operation.type === 'delete') {
                        const token = pythonTokens[operation.indexA];
                        if (token && !(token.type === 'text' && token.isWhitespace)) {
                            pythonHighlights[operation.indexA] = 'added';
                        }
                    } else if (operation.type === 'insert') {
                        const token = tsTokens[operation.indexB];
                        if (token && !(token.type === 'text' && token.isWhitespace)) {
                            tsHighlights[operation.indexB] = 'removed';
                        }
                    }
                });

                return {
                    python: wrapCodeContent(renderTokens(pythonTokens, pythonHighlights), mode),
                    typescript: wrapCodeContent(renderTokens(tsTokens, tsHighlights), mode),
                };
            }

            const { pythonHighlights, tsHighlights } = buildCharDiffHighlights(pythonTokens, tsTokens, operations);

            return {
                python: wrapCodeContent(renderTokens(pythonTokens, pythonHighlights), mode),
                typescript: wrapCodeContent(renderTokens(tsTokens, tsHighlights), mode),
            };
        }

        function tokenizeHtml(html) {
            if (!html) {
                return [];
            }
            const tokens = [];
            const tagRegex = /<[^>]+?>/g;
            let lastIndex = 0;
            let match;
            while ((match = tagRegex.exec(html)) !== null) {
                if (match.index > lastIndex) {
                    const textChunk = html.slice(lastIndex, match.index);
                    appendTextTokens(tokens, textChunk);
                }
                tokens.push(createTagToken(match[0]));
                lastIndex = match.index + match[0].length;
            }
            if (lastIndex < html.length) {
                const remainder = html.slice(lastIndex);
                appendTextTokens(tokens, remainder);
            }
            return tokens;
        }

        function appendTextTokens(target, text) {
            if (!text) {
                return;
            }
            let splitter = unicodeAwareSplitter;
            if (!splitter) {
                splitter = fallbackSplitter;
            }
            splitter.lastIndex = 0;
            let lastIndex = 0;
            let match;
            while ((match = splitter.exec(text)) !== null) {
                if (match.index > lastIndex) {
                    target.push(createTextToken(text.slice(lastIndex, match.index)));
                }
                target.push(createTextToken(match[0]));
                lastIndex = splitter.lastIndex;
            }
            if (lastIndex < text.length) {
                target.push(createTextToken(text.slice(lastIndex)));
            }
        }

        let unicodeAwareSplitter = null;
        let fallbackSplitter = /(\s+|[^\w]+)/g;
        try {
            unicodeAwareSplitter = new RegExp('([\\s]+|[^\\p{L}\\p{N}]+)', 'gu');
        } catch (error) {
            unicodeAwareSplitter = null;
            fallbackSplitter = /(\s+|[^A-Za-z0-9_]+)/g;
        }

        function createTextToken(raw) {
            return {
                type: 'text',
                raw,
                isWhitespace: /^\s+$/.test(raw),
            };
        }

        function createTagToken(raw) {
            const isClosing = /^<\s*\//.test(raw);
            const tagNameMatch = raw.match(/^<\s*\/?\s*([a-zA-Z0-9:-]+)/);
            return {
                type: 'tag',
                raw,
                isClosing,
                tagName: tagNameMatch ? tagNameMatch[1].toLowerCase() : "",
            };
        }

        function tokensEqual(leftToken, rightToken) {
            if (!leftToken || !rightToken || leftToken.type !== rightToken.type) {
                return false;
            }
            return leftToken.raw === rightToken.raw;
        }

        function diffUsingLcs(leftTokens, rightTokens, comparator) {
            const n = leftTokens.length;
            const m = rightTokens.length;
            if (!n && !m) {
                return [];
            }

            const table = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
            for (let i = n - 1; i >= 0; i -= 1) {
                for (let j = m - 1; j >= 0; j -= 1) {
                    if (comparator(leftTokens[i], rightTokens[j])) {
                        table[i][j] = table[i + 1][j + 1] + 1;
                    } else {
                        table[i][j] = Math.max(table[i + 1][j], table[i][j + 1]);
                    }
                }
            }

            const operations = [];
            let i = 0;
            let j = 0;
            while (i < n && j < m) {
                if (comparator(leftTokens[i], rightTokens[j])) {
                    operations.push({ type: 'equal', indexA: i, indexB: j });
                    i += 1;
                    j += 1;
                } else if (table[i + 1][j] >= table[i][j + 1]) {
                    operations.push({ type: 'delete', indexA: i });
                    i += 1;
                } else {
                    operations.push({ type: 'insert', indexB: j });
                    j += 1;
                }
            }

            while (i < n) {
                operations.push({ type: 'delete', indexA: i });
                i += 1;
            }
            while (j < m) {
                operations.push({ type: 'insert', indexB: j });
                j += 1;
            }

            return operations;
        }

        function buildCharDiffHighlights(pythonTokens, tsTokens, operations) {
            const pythonHighlights = new Array(pythonTokens.length).fill(null);
            const tsHighlights = new Array(tsTokens.length).fill(null);

            let cursor = 0;
            while (cursor < operations.length) {
                const op = operations[cursor];
                if (op.type === 'equal') {
                    cursor += 1;
                    continue;
                }

                const deleteBatch = [];
                const insertBatch = [];

                while (cursor < operations.length && operations[cursor].type === 'delete') {
                    deleteBatch.push(operations[cursor]);
                    cursor += 1;
                }
                while (cursor < operations.length && operations[cursor].type === 'insert') {
                    insertBatch.push(operations[cursor]);
                    cursor += 1;
                }

                if (!deleteBatch.length && !insertBatch.length) {
                    cursor += 1;
                    continue;
                }

                if (deleteBatch.length === 1 && insertBatch.length === 1) {
                    const deleteToken = pythonTokens[deleteBatch[0].indexA];
                    const insertToken = tsTokens[insertBatch[0].indexB];
                    if (deleteToken && insertToken && deleteToken.type === 'text' && insertToken.type === 'text' && !deleteToken.isWhitespace && !insertToken.isWhitespace) {
                        const charDiff = diffChars(deleteToken.raw, insertToken.raw);
                        pythonHighlights[deleteBatch[0].indexA] = { segments: charDiff.pythonSegments };
                        tsHighlights[insertBatch[0].indexB] = { segments: charDiff.tsSegments };
                        continue;
                    }
                }

                deleteBatch.forEach((entry) => {
                    const token = pythonTokens[entry.indexA];
                    if (token && !(token.type === 'text' && token.isWhitespace)) {
                        pythonHighlights[entry.indexA] = 'added';
                    }
                });
                insertBatch.forEach((entry) => {
                    const token = tsTokens[entry.indexB];
                    if (token && !(token.type === 'text' && token.isWhitespace)) {
                        tsHighlights[entry.indexB] = 'removed';
                    }
                });
            }

            return { pythonHighlights, tsHighlights };
        }

        function diffChars(leftText, rightText) {
            const leftChars = Array.from(leftText || "");
            const rightChars = Array.from(rightText || "");
            const operations = diffUsingLcs(leftChars, rightChars, (a, b) => a === b);
            const pythonSegments = [];
            const tsSegments = [];

            operations.forEach((operation) => {
                if (operation.type === 'equal') {
                    appendSegment(pythonSegments, leftChars[operation.indexA], null);
                    appendSegment(tsSegments, rightChars[operation.indexB], null);
                } else if (operation.type === 'delete') {
                    appendSegment(pythonSegments, leftChars[operation.indexA], 'added');
                } else if (operation.type === 'insert') {
                    appendSegment(tsSegments, rightChars[operation.indexB], 'removed');
                }
            });

            return { pythonSegments, tsSegments };
        }

        function appendSegment(target, text, highlight) {
            if (!text) {
                return;
            }
            const last = target[target.length - 1];
            if (last && last.highlight === highlight) {
                last.text += text;
            } else {
                target.push({ text, highlight });
            }
        }

        function renderTokens(tokens, highlights) {
            if (!tokens.length) {
                return "";
            }
            const parts = [];
            for (let index = 0; index < tokens.length; index += 1) {
                const token = tokens[index];
                const highlight = highlights ? highlights[index] : null;

                if (highlight && typeof highlight === 'object' && Array.isArray(highlight.segments)) {
                    parts.push(renderSegments(highlight.segments));
                    continue;
                }

                const tokenText = escapeHtml(token.raw);

                if (highlight === 'added' || highlight === 'removed') {
                    if (token.type === 'text' && token.isWhitespace) {
                        parts.push(tokenText);
                    } else {
                        const className = highlight === 'added' ? 'diff-added' : 'diff-removed';
                        parts.push(`<span class="${className}">${tokenText}</span>`);
                    }
                } else {
                    parts.push(tokenText);
                }
            }
            return parts.join("");
        }

        function renderSegments(segments) {
            return segments.map((segment) => {
                const safeText = escapeHtml(segment.text || "");
                if (!segment.highlight) {
                    return safeText;
                }
                const className = segment.highlight === 'added' ? 'diff-added' : 'diff-removed';
                return `<span class="${className}">${safeText}</span>`;
            }).join("");
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

            setupPageNumberJump();
            initializeThumbnailDrawer();
            initializeDiffControls();
            initializeAgentUI();

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
            syncThumbnailDrawerHeight();
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
            try:
                result = run_agent_via_responses(agent, request_payload)
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
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), ComparisonHandler) as httpd:
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
