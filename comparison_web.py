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
import html
import re
from pathlib import Path

# Import původního procesoru
from main_processor import AltoProcessor


ROOT_DIR = Path(__file__).resolve().parent
TS_DIST_PATH = ROOT_DIR / 'dist' / 'run_original.js'

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
        .results {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .action-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
        }
        .preview-wrapper {
            position: relative;
        }
        .preview-wrapper button {
            background-color: #6c757d;
        }
        .preview-wrapper button:hover {
            background-color: #545b62;
        }
        .preview-container {
            display: none;
            position: absolute;
            top: calc(100% + 2px);
            right: 0;
            text-align: center;
            background: white;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            max-width: min(600px, 45vw);
            z-index: 10;
        }
        .preview-container img {
            max-width: min(580px, 42vw);
            max-height: min(800px, 80vh);
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
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
            text-align: center;
            padding: 20px;
            display: none;
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
        pre {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ALTO Processing Comparison</h1>
        <p>Porovnání původního TypeScript a nového Python zpracování ALTO XML</p>

        <div class="form-group">
            <label for="uuid">UUID dokumentu:</label>
            <input type="text" id="uuid" placeholder="Zadejte UUID (např. 673320dd-0071-4a03-bf82-243ee206bc0b)" value="673320dd-0071-4a03-bf82-243ee206bc0b">
        </div>

        <div class="form-group">
            <label for="width">Šířka:</label>
            <input type="number" id="width" value="800">
        </div>

        <div class="form-group">
            <label for="height">Výška:</label>
            <input type="number" id="height" value="1200">
        </div>

        <div class="action-row">
            <button onclick="processAlto()">Zpracovat a porovnat</button>
            <div class="preview-wrapper" onmouseenter="showPreview()" onmouseleave="hidePreview()">
                <button id="preview-button" type="button">Náhled stránky</button>
                <div id="preview-container" class="preview-container">
                    <p id="preview-status" style="margin-bottom: 10px; color: #555;"></p>
                    <img id="preview-image" alt="Náhled stránky">
                </div>
            </div>
        </div>

        <div id="loading" class="loading">
            <p>Zpracovávám ALTO data...</p>
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
        let previewVisible = false;
        let previewObjectUrl = null;
        let previewFetching = false;
        let previewImageUuid = null;

        function resetPreview() {
            const container = document.getElementById('preview-container');
            const img = document.getElementById('preview-image');
            const status = document.getElementById('preview-status');

            if (previewObjectUrl) {
                URL.revokeObjectURL(previewObjectUrl);
                previewObjectUrl = null;
            }

            previewImageUuid = null;
            previewFetching = false;
            previewVisible = false;

            if (img) {
                img.src = '';
            }
            if (status) {
                status.textContent = '';
            }
            if (container) {
                container.style.display = 'none';
            }
        }

        async function showPreview() {
            const uuid = document.getElementById('uuid').value;
            const container = document.getElementById('preview-container');
            const img = document.getElementById('preview-image');
            const status = document.getElementById('preview-status');

            if (!container || !img || !status) {
                return;
            }

            if (!uuid) {
                status.textContent = 'Zadejte UUID';
                container.style.display = 'block';
                previewVisible = true;
                return;
            }

            previewVisible = true;
            container.style.display = 'block';

            if (previewObjectUrl && previewImageUuid !== uuid) {
                URL.revokeObjectURL(previewObjectUrl);
                previewObjectUrl = null;
                previewImageUuid = null;
            }

            if (previewObjectUrl) {
                img.src = previewObjectUrl;
                status.textContent = '';
                return;
            }

            if (previewFetching) {
                status.textContent = 'Načítám náhled...';
                return;
            }

            previewFetching = true;
            status.textContent = 'Načítám náhled...';

            const requestUuid = uuid;

            try {
                const response = await fetch(`/preview?uuid=${encodeURIComponent(uuid)}&stream=IMG_FULL`, { cache: 'no-store' });
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const blob = await response.blob();

                const currentUuid = document.getElementById('uuid').value;
                if (!previewVisible || currentUuid !== requestUuid) {
                    return;
                }

                if (previewObjectUrl) {
                    URL.revokeObjectURL(previewObjectUrl);
                }
                previewObjectUrl = URL.createObjectURL(blob);
                previewImageUuid = requestUuid;

                img.src = previewObjectUrl;
                img.alt = 'Náhled stránky';
                status.textContent = '';
            } catch (error) {
                console.error('Chyba při načítání náhledu:', error);
                status.textContent = 'Náhled se nepodařilo načíst.';
            } finally {
                previewFetching = false;
            }
        }

        function hidePreview() {
            const container = document.getElementById('preview-container');
            if (container) {
                container.style.display = 'none';
            }
            previewVisible = false;
        }

        async function processAlto() {
            const uuid = document.getElementById('uuid').value;
            const width = document.getElementById('width').value;
            const height = document.getElementById('height').value;

            if (!uuid) {
                alert('Zadejte UUID');
                return;
            }

            resetPreview();
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';

            try {
                const response = await fetch(`/process?uuid=${encodeURIComponent(uuid)}&width=${width}&height=${height}`);
                const data = await response.json();

                document.getElementById('loading').style.display = 'none';

                if (data.error) {
                    alert('Chyba: ' + data.error);
                    return;
                }

                document.getElementById('python-result').innerHTML = '<pre>' + data.python + '</pre>';
                document.getElementById('typescript-result').innerHTML = '<pre>' + data.typescript + '</pre>';
                document.getElementById('results').style.display = 'grid';

            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Chyba při zpracování: ' + error);
            }
        }

        // Automatické zpracování při načtení stránky
        window.onload = function() {
            processAlto();
        };
    </script>
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
            width = int(query_params.get('width', ['800'])[0])
            height = int(query_params.get('height', ['1200'])[0])

            if not uuid:
                self.wfile.write(json.dumps({'error': 'UUID je povinný'}).encode('utf-8'))
                return

            try:
                # Python zpracování
                processor = AltoProcessor()
                alto_xml = processor.get_alto_data(uuid)

                if not alto_xml:
                    self.wfile.write(json.dumps({'error': 'Nepodařilo se stáhnout ALTO data'}).encode('utf-8'))
                    return

                python_result = processor.get_formatted_text(alto_xml, uuid, width, height)

                # Simulace TypeScript výsledku (pro demonstraci)
                # V reálném scénáři byste zavolali původní TypeScript službu
                typescript_result = simulate_typescript_processing(alto_xml, uuid, width, height)

                response_data = {
                    'python': python_result,
                    'typescript': typescript_result
                }

                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))

            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

        elif self.path.startswith('/preview'):
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            uuid = query_params.get('uuid', [''])[0]
            stream = query_params.get('stream', ['IMG_FULL'])[0]
            allowed_streams = {'IMG_THUMB', 'IMG_PREVIEW', 'IMG_FULL'}

            if not uuid:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'UUID je povinný'}).encode('utf-8'))
                return

            if stream not in allowed_streams:
                stream = 'IMG_FULL'

            upstream_url = f"https://kramerius5.nkp.cz/search/api/v5.0/item/uuid:{uuid}/streams/{stream}"

            try:
                response = requests.get(upstream_url, timeout=20)
                if response.status_code != 200 or not response.content:
                    self.send_response(502)
                    self.send_header('Content-type', 'application/json; charset=utf-8')
                    self.end_headers()
                    message = f'Nepodařilo se načíst náhled (status {response.status_code})'
                    self.wfile.write(json.dumps({'error': message}).encode('utf-8'))
                    return

                content_type = response.headers.get('Content-Type', 'image/jpeg')
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-Length', str(len(response.content)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(response.content)

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
