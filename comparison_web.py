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


ROOT_DIR = Path(__file__).resolve().parent
TS_DIST_PATH = ROOT_DIR / 'dist' / 'run_original.js'
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 1200

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
        if self.path == '/':
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
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            position: relative;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
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
        .loading {
            position: absolute;
            inset: 0;
            background: rgba(255, 255, 255, 0.82);
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 12px;
            text-align: center;
            z-index: 100;
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
    <div class="container">
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

        <div id="loading" class="loading" aria-live="polite" aria-hidden="true">
            <div class="loading-content">
                <div class="loading-spinner" role="presentation"></div>
                <p>Zpracovávám ALTO data...</p>
            </div>
        </div>

        <div id="results" class="results" style="display: none;">
            <div class="result-box">
                <h3>Python výsledek</h3>
                <div id="python-result"></div>
            </div>
            <div class="result-box">
                <h3>TypeScript výsledek (simulace)</h3>
                <div id="typescript-result"></div>
            </div>
        </div>
    </div>

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
        let currentAltoXml = '';

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
                const response = await fetch(`/process?uuid=${encodeURIComponent(uuid)}`, { cache: "no-store" });
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
            const status = document.getElementById("preview-status");

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

            if (status) {
                status.textContent = "";
                status.style.display = "none";
            }

            if (container) {
                container.style.display = "none";
                container.classList.remove("preview-visible", "preview-loaded", "preview-error");
                delete container.dataset.previewStream;
            }
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
            const status = document.getElementById("preview-status");

            if (!container || !thumb || !largeImg || !largeBox || !status || !previewObjectUrl) {
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
                    status.textContent = "Náhled se nepodařilo načíst.";
                    status.style.display = "block";
                    container.classList.add("preview-error");
                    container.classList.remove("preview-loaded");
                    setLargePreviewActive(false);
                };
                if (thumb.complete && thumb.naturalWidth > 0) {
                    finalize();
                } else if (thumb.complete) {
                    status.textContent = "Náhled se nepodařilo načíst.";
                    status.style.display = "block";
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
                    status.textContent = "Náhled se nepodařilo načíst.";
                    status.style.display = "block";
                    container.classList.add("preview-error");
                    container.classList.remove("preview-loaded");
                    setLargePreviewActive(false);
                };
            }

            container.style.display = "flex";
            container.classList.add("preview-visible", "preview-loaded");
            container.classList.remove("preview-error");
            status.textContent = "";
            status.style.display = "none";

            setLargePreviewActive();
        }

        async function loadPreview(uuid) {
            const container = document.getElementById("page-preview");
            const thumb = document.getElementById("preview-image-thumb");
            const largeImg = document.getElementById("preview-image-large");
            const largeBox = document.getElementById("preview-large");
            const status = document.getElementById("preview-status");

            if (!container || !thumb || !largeImg || !largeBox || !status || !uuid) {
                return;
            }

            setLargePreviewActive(false);

            if (previewImageUuid === uuid && previewObjectUrl) {
                showPreviewFromCache();
                return;
            }

            const cachedEntry = previewCache.get(uuid);
            if (cachedEntry && cachedEntry.objectUrl) {
                previewImageUuid = uuid;
                previewObjectUrl = cachedEntry.objectUrl;
                showPreviewFromCache(cachedEntry);
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
            status.textContent = "Načítám náhled...";
            status.style.display = "block";
            thumb.style.display = "block";
            thumb.style.opacity = "0";

            let handleLoad;

            try {
                const entry = await ensurePreviewEntry(uuid);

                if (!entry || previewImageUuid !== uuid) {
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
                        status.textContent = "Náhled se nepodařilo načíst.";
                        status.style.display = "block";
                        container.classList.add("preview-error");
                        container.classList.remove("preview-loaded");
                        setLargePreviewActive();
                    }
                };
                thumb.src = previewObjectUrl;
                thumb.style.opacity = "1";

                if (thumb.complete && thumb.naturalWidth === 0) {
                    if (previewImageUuid === uuid) {
                        status.textContent = "Náhled se nepodařilo načíst.";
                        status.style.display = "block";
                        container.classList.add("preview-error");
                        container.classList.remove("preview-loaded");
                        setLargePreviewActive();
                    }
                }

                container.dataset.previewStream = entry.stream;

                container.classList.add("preview-loaded");
                status.textContent = "";
                status.style.display = "none";

                setLargePreviewActive();
            } catch (error) {
                if (previewImageUuid === uuid) {
                    console.error("Chyba při načítání náhledu:", error);
                    status.textContent = "Náhled se nepodařilo načíst.";
                    status.style.display = "block";
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
            const uuidField = document.getElementById("uuid");
            if (uuidField) {
                uuidField.value = targetUuid;
            }
            processAlto();
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

            currentAltoXml = data.alto_xml || '';
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

            if (currentPage && currentPage.uuid) {
                loadPreview(currentPage.uuid);
            } else {
                resetPreview();
            }

            const pythonResult = document.getElementById("python-result");
            if (pythonResult) {
                pythonResult.innerHTML = `<pre>${data.python || ""}</pre>`;
            }

            const tsResult = document.getElementById("typescript-result");
            if (tsResult) {
                tsResult.innerHTML = `<pre>${data.typescript || ""}</pre>`;
            }

            const results = document.getElementById("results");
            if (results) {
                results.style.display = "grid";
            }

            const uuidField = document.getElementById("uuid");
            if (uuidField && currentPage && currentPage.uuid) {
                uuidField.value = currentPage.uuid;
            }

            updateCacheWindow(currentPage ? currentPage.uuid : uuid, data.navigation || null);
            schedulePrefetch(data.navigation || null);
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
                        modalContent.style.top = '';
                        modalContent.style.left = '';
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
            self.wfile.write(html.encode('utf-8'))

        elif self.path.startswith('/process'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()

            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            uuid = query_params.get('uuid', [''])[0]

            if not uuid:
                self.wfile.write(json.dumps({'error': 'UUID je povinný'}).encode('utf-8'))
                return

            try:
                processor = AltoProcessor()
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
                for base in candidate_bases:
                    for candidate in candidate_streams:
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
                            if stream == 'AUTO':
                                continue
                            break

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
            self.wfile.write(b'404 - Nenalezeno')

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
