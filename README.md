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

## Uživatelské API / UI

- Frontend visí na `/` a komunikace probíhá přes REST (`/process`, `/preview`, `/diff`, `/agents/*`, `/exports/*`).  
- `/healthz` je vhodný pro load balancer / monitoring.  
- Agenti jsou ukládáni jako JSON soubory ve složkách `agents/` (`correctors`, `joiners`, `readers`).  
- Konfigurace modelů je v `config/models.json` a při běhu se načítá automaticky.

## Tipy

- Node.js je pouze kvůli legacy TypeScriptu (`dist/run_original.js`). Jakmile nebude potřeba, lze kroky s NPM z Dockerfile i lokální instalace úplně odstranit.  
- `WEB_CONCURRENCY` ve `start.sh` nastavte na počet CPU jáder (např. 4).  
- Exportní joby běží ve vlákně procesu a výstupy ukládají do dočasných souborů; restart procesu je smaže.  
- Pokud re-runnete kontejner bez volume, přijdete o uložené agenty i výstupy – ujistěte se, že `agents/` a `config/` jsou bindnuté.  
- Server musí mít odchozí HTTPS k API Krameria (MZK, NKP…) a k OpenRouter/OpenAI API; bez toho neproběhne procesování ani agentí volání.
