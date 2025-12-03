# Alto Processing Web Service

Tento repozitář obsahuje FastAPI verzi porovnávacího webu. UI je stejné jako ve skriptu `comparison_web.py` – renderuje šablonu `app/templates/compare.html` a komunikuje přes REST endpointy `/process`, `/preview`, `/diff`, `/agents/*`, `/exports/*`, které běží ve stejném procesu.

## Prerekvizity

- Python 3.10+
- Node.js + npm (stačí kvůli legacy TypeScript bundlu pro simulaci původního kódu)
- Docker + docker compose plugin (pokud chcete deployovat kontejnerem)

## Lokální spuštění (bez Dockeru)

1. Naklonujte repo a pracujte v jeho kořenové složce.  
2. Zkopírujte `.env.example` na `.env` a doplňte API klíče (OpenRouter/OpenAI) a další proměnné. Soubor `.env` je v `.gitignore`, takže se do gitu necommitne.  
3. Vytvořte virtuální prostředí a nainstalujte závislosti:
   ```bash
   python3 -m venv .webenv
   source .webenv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Nainstalujte Node závislosti a (volitelně) přebuilděte TypeScript:
   ```bash
   npm install
   # volitelné – ruční rebuild bundlu pro legacy procesor:
   npx tsc
   ```
5. Spusťte server:
   ```bash
   ./start.sh  # respektuje HOST, PORT, WEB_CONCURRENCY, LOG_LEVEL
   # nebo ručně:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8080 --app-dir .
   ```

Po spuštění otevřete http://localhost:8080. Endpoint `/healthz` vrací JSON `{"status": "ok", "environment": "<hodnota ALTO_WEB_ENVIRONMENT>"}`.

## Struktura projektu

- `app/` – FastAPI aplikace (`app/main.py`), šablony UI (`app/templates/compare.html`), aplikační logika v `app/core/`.
- `static/` – statická aktiva UI.
- `agents/`, `config/` – runtime data ukládaná při běhu; musí být uchovaná přes volume.
- `dist/` – výsledný JS bundle (`run_original.js`) generovaný TypeScriptem.
- `start.sh` – wrapper nad uvicornem, respektuje proměnné `HOST`, `PORT`, `WEB_CONCURRENCY`, `LOG_LEVEL`.

## Docker image

`Dockerfile` v kořeni:

- nainstaluje systémové Node.js + npm (kvůli TypeScriptu),
- nainstaluje Python závislosti (`requirements.txt`),
- provede `npm ci && npx tsc`, aby byl k dispozici `dist/run_original.js`,
- spustí uvicorn (`app.main:app`) na portu 8080.

Build a spuštění v kořenové složce:

```bash
docker build -t alto-web .
docker run --rm -p 8080:8080 --env-file .env alto-web
```

Environment proměnné aplikace začínají prefixem `ALTO_WEB_`. Soubor `.env` obsahuje jak FastAPI proměnné, tak klíče pro agenty – díky `python-dotenv` se načte při startu.

### Docker Compose

Pohodlnější je `docker compose` v kořenové složce:

```bash
cp .env.example .env   # doplňte hodnoty
docker compose up --build -d
```

Služba mapuje na hostiteli složky `./agents` a `./config`, aby uložené agenty a konfigurace zůstaly zachované i po restartu kontejneru. Pokud používáte čisté `docker run`, přidejte `-v $(pwd)/agents:/app/agents -v $(pwd)/config:/app/config`, jinak se úpravy agentů ztratí s kontejnerem. Port 8080 je zveřejněný navenek (`8080:8080`).

### Nasazení s Nginx reverse proxy

Aktuální Compose soubor vystavuje FastAPI kontejner přímo na port 80 (host) → 8080 (container). Pro produkční HTTPS provoz je doporučené přidat vstupní Nginx vrstvu:

1. **Proxy kontejner** – doplňte Compose službu `nginx` (nebo Traefik/Caddy). Naslouchá na hostitelských portech 80 a 443, sdílí síť s `alto-web` a předává requesty na `http://alto-web:8080`.
2. **TLS certifikáty** – používejte Let’s Encrypt (HTTP-01 challenge na portu 80). Certy ukládejte do bind mountu (např. `./certs:/etc/letsencrypt`). Nginx může používat `certbot` kontejner, případně cron na hostu.
3. **DNS** – doména `alto-processing.trinera.cloud` už míří na server. Pokud přibudou subdomény (např. `api.alto-processing...`), vytvořte odpovídající DNS záznamy před generováním certifikátů.
4. **Konfigurace** – doporučené je vytvořit složku `deploy/` s `nginx.conf` a případně `docker-compose.override.yml`, kde budou definované proxy služby, volume s certy a mapování portů. Aplikace nevyžaduje žádné změny kromě toho, že Nginx by měl posílat hlavičky `Host`, `X-Forwarded-For` a `X-Forwarded-Proto`. FastAPI zvládne obsluhovat více hostnames – pokud se rozhodnete oddělit například API na subdoménu (`api.alto-processing...`), stačí v proxy přidat další `server` blok s `proxy_pass` na stejný backend.
5. **Testování** – po spuštění proxy ověřte `https://alto-processing.trinera.cloud/healthz`, UI na `/` a REST endpointy. Jakmile HTTPS funguje, zapněte redirect z HTTP na HTTPS (301) a případně HSTS.

Dočasně může HTTP a HTTPS běžet souběžně (např. HTTP jen pro Let’s Encrypt challenge). Jakmile je reverse proxy stabilní, není nutné vystavovat port 8080/80 přímo z FastAPI kontejneru.

## Nginx reverse proxy (Compose override)

Repo obsahuje `deploy/docker-compose.override.yml` + `deploy/nginx.conf`, které přidají Nginx před `alto-web`:

1. Certy necháváme na hostu v `/etc/letsencrypt` (standardní umístění certbotu). Nginx si je mountuje read-only.
   - HTTP-01 challenge se servíruje z `/var/www/certbot/.well-known/` (certbot tam ukládá soubory).
2. Spusť: `docker compose -f docker-compose.yml -f deploy/docker-compose.override.yml up --build -d`
3. Ověř: `https://alto-processing.trinera.cloud/healthz`

Lokální test bez certů (jen HTTP pass-through): místo override spusť `docker compose up --build -d` (poběží na portu 80 → 8080 bez TLS). Pro HTTPS lokálně bys musel vytvořit self-signed certy a uložit je na stejné cesty, které emulují `/etc/letsencrypt`.

## Uživatelské API / UI

- Frontend visí na `/` a komunikace probíhá přes REST (`/process`, `/preview`, `/diff`, `/agents/*`, `/exports/*`).  
- `/healthz` je vhodný pro load balancer / monitoring.  
- Agenti jsou ukládáni jako JSON soubory ve složkách `agents/` (`correctors`, `joiners`, `readers`).  
- Konfigurace modelů je v `config/models.json` a při běhu se načítá automaticky.

### Jednoduchý download přes API

Endpoint `POST /download` spustí export a vrátí `job_id`. Stav zjistíš přes `GET /exports/{job_id}` a výsledek stáhneš z `GET /exports/{job_id}/download`.

Auth: pošli token v hlavičce `Authorization: Bearer <ALTO_WEB_AUTH_TOKEN>`.

Body (JSON):
- `uuid` (povinné): UUID knihy nebo stránky.
- `format` (volitelné): `txt` (default), `html`, `md`.
- `range` (volitelné): `all` (celá kniha), nebo vlastní výběr např. `"7-11,23"`. Pokud chybí: u stránky se vezme jen ta stránka, u knihy celé.
- `llmAgent` (volitelné): např. `{ "name": "cleanup-diff-generated-mid" }`. Pokud chybí, použije se čistě algoritmický výstup bez LLM.
- `dropSmall` (volitelné): `true/false`, default `false`.
- `outputName` (volitelné): název výsledného souboru.

Praktický příklad (celá kniha, HTML, algoritmus):
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
Vlastní rozsah + LLM + dropSmall:
```bash
TOKEN="..."; BASE="http://localhost:8080"
curl -X POST "$BASE/download" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "uuid": "49c6424a-c820-4224-9475-4aa0d8a9d844",
    "format": "txt",
    "range": "7-11,23",
    "llmAgent": { "name": "cleanup-diff-generated-mid" },
    "dropSmall": true,
    "outputName": "output.txt"
  }'
```
Polling a download:
```bash
curl -H "Authorization: Bearer $TOKEN" "$BASE/exports/<job_id>"
curl -H "Authorization: Bearer $TOKEN" -o output.txt "$BASE/exports/<job_id>/download"
```

### CLI helper

Soubor `cli/download.py` dělá totéž: spustí download, sleduje stav a uloží výsledek. Token bere z argumentu `--token` nebo z proměnné `ALTO_TOKEN`.
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

## Tipy

- Node.js je pouze kvůli legacy TypeScriptu (`dist/run_original.js`). Jakmile nebude potřeba, lze kroky s NPM z Dockerfile i lokální instalace úplně odstranit.  
- `WEB_CONCURRENCY` ve `start.sh` nastavte na počet CPU jáder (např. 4).  
- Exportní joby běží ve vlákně procesu a výstupy ukládají do dočasných souborů; restart procesu je smaže.  
- Pokud re-runnete kontejner bez volume, přijdete o uložené agenty i výstupy – ujistěte se, že `agents/` a `config/` jsou bindnuté.  
- Server musí mít odchozí HTTPS k API Krameria (MZK, NKP…) a k OpenRouter/OpenAI API; bez toho neproběhne procesování ani agentí volání.
