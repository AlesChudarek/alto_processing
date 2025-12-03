# Alto Processing Web Service

FastAPI aplikace s UI (šablona `app/templates/compare.html`) a REST endpointy `/process`, `/preview`, `/diff`, `/agents/*`, `/exports/*`. Běží jako jeden proces; UI komunikuje přes REST.

## Co potřebuješ
- Python 3.10+
- Node.js + npm (jen kvůli legacy TypeScript bundlu `dist/run_original.js`)
- Docker + Docker Compose plugin (pro kontejnerové spuštění)

## Rychlý start lokálně (bez Dockeru)
1) `cp .env.example .env` a doplň API klíče (OpenRouter/OpenAI) + další proměnné.  
2) Virtuální env a závislosti:
```bash
python3 -m venv .webenv
source .webenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
3) Node závislosti a případný rebuild TS:
```bash
npm install
npx tsc   # jen pokud potřebuješ přebuildit dist/run_original.js
```
4) Spuštění:
```bash
./start.sh  # respektuje HOST, PORT, WEB_CONCURRENCY, LOG_LEVEL
# nebo:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080 --app-dir .
```
Otevři http://localhost:8080. `/healthz` vrací `{"status": "ok", "environment": "<ALTO_WEB_ENVIRONMENT>"}`.

## Docker / Compose
- Build a run:
```bash
docker build -t alto-web .
docker run --rm -p 8080:8080 --env-file .env alto-web
```
- Compose ( pohodlnější, bez port mappingu v defaultu ):
```bash
cp .env.example .env
docker compose up --build -d
```
Kontejner mapuje `./agents` a `./config` jako bind mounty. Port 8080 je dostupný jen v síti Compose; do světa ho publikuje Nginx v override (viz níže).

## Produkce: Nginx reverse proxy + Let’s Encrypt
Repo má připravené `deploy/docker-compose.override.yml` (služba `nginx`) a `deploy/nginx.conf`.

1) Certy Let’s Encrypt na hostu (certbot webroot):
```bash
sudo mkdir -p /var/www/certbot/.well-known/acme-challenge
sudo certbot certonly --webroot -w /var/www/certbot \
  -d alto-processing.trinera.cloud --agree-tos -m tvuj@email.cz --non-interactive
```
2) Spuštění s proxy:
```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.override.yml up --build -d
```
Nginx vystaví 80/443, předává na `alto-web:8080`, posílá hlavičky Host/X-Forwarded-For/Proto, přesměrovává HTTP→HTTPS, HSTS je zapnuté.
3) Ověření: `https://alto-processing.trinera.cloud/healthz` (200), `curl -I http://.../healthz` (301 na https).
4) Obnova certů (cron pro root, 3:00 denně):
```
0 3 * * * certbot renew --webroot -w /var/www/certbot --post-hook "docker compose -f /root/alto_processing/docker-compose.yml -f /root/alto_processing/deploy/docker-compose.override.yml exec nginx nginx -s reload"
```

## Struktura
- `app/` FastAPI (`app/main.py`), šablony `app/templates/compare.html`, logika v `app/core/`
- `static/` statická aktiva UI
- `agents/`, `config/` runtime data (bind mount v Compose)
- `dist/` bundlovaný legacy JS `run_original.js` (build přes `npx tsc`)
- `start.sh` wrapper nad uvicorn (čte HOST, PORT, WEB_CONCURRENCY, LOG_LEVEL)

## API / UI
- UI na `/`, REST `/process`, `/preview`, `/diff`, `/agents/*`, `/exports/*`.
- `/healthz` pro monitoring.
- Token v hlavičce `Authorization: Bearer <ALTO_WEB_AUTH_TOKEN>` (healthz je veřejné, chráněné endpointy vrací 303/401 bez tokenu).

### Download přes API
`POST /download` → vrátí `job_id`. Sleduj `GET /exports/{job_id}`, stáhni `GET /exports/{job_id}/download`.
Tělo (JSON): `uuid` (povinné), `format` (`txt` default/`html`/`md`), `range` (`all` nebo např. `"7-11,23"`), `llmAgent` (např. `{ "name": "cleanup-diff-generated-mid" }`), `dropSmall` (bool), `outputName`.

Příklad (lokálně):
```bash
TOKEN="..."; BASE="http://localhost:8080"
curl -X POST "$BASE/download" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "uuid": "49c6424a-c820-4224-9475-4aa0d8a9d844",
    "format": "html",
    "range": "all",
    "outputName": "output.html"
  }'
```
Polling a stažení:
```bash
curl -H "Authorization: Bearer $TOKEN" "$BASE/exports/<job_id>"
curl -H "Authorization: Bearer $TOKEN" -o output.txt "$BASE/exports/<job_id>/download"
```

### CLI helper
`cli/download.py` dělá totéž; token z `--token` nebo `ALTO_TOKEN`.
```bash
ALTO_TOKEN=TVŮJ_TOKEN python cli/download.py \
  --url http://localhost:8080 \
  --uuid 49c6424a-c820-4224-9475-4aa0d8a9d844 \
  --format txt \
  --range "7-11,23" \
  --llm-agent '{"name":"cleanup-diff-generated-mid"}' \
  --drop-small \
  --output output.txt
```

## Poznámky
- Node.js je jen kvůli legacy TypeScriptu; pokud bundl nebude potřeba, lze NPM kroky z Dockerfile odstranit.
- `WEB_CONCURRENCY` ve `start.sh` nastav na počet CPU jader.
- Exportní joby ukládají do dočasných souborů; při restartu zmizí. Bind mounty `agents/` a `config/` si drž ve storage.
- Server musí mít odchozí HTTPS k API Krameria a OpenRouter/OpenAI; jinak zpracování padne.
