# Alto Processing Web Service

Tahle složka obsahuje FastAPI verzi původního porovnávacího webu. UI je renderované identicky jako ve skriptu `comparison_web.py` a komunikuje přes REST endpoints (`/process`, `/preview`, `/diff`, `/agents/*`) běžící uvnitř stejného kontejneru.

## Lokální spuštění

1. `cd web_service` – všechny příkazy spouštějte z této složky (je to budoucí samostatné repo).  
2. Zkopírujte `.env.example` na `.env` a doplňte API klíče (OpenRouter/OpenAI).  
3. Vytvořte izolované prostředí:
   ```bash
   python3 -m venv .webenv
   source .webenv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Nainstalujte Node závislosti (pro TypeScript simulaci původního kódu):
   ```bash
   npm install
   # (volitelně) npx tsc  # buildne dist/run_original.js, jinak se přebuildí při prvním použití
   ```
5. Spusťte server (můžete použít i připravený skript, který řídí počet workerů):
   ```bash
   ./start.sh  # používá uvicorn, defaultně 2 workery
   # nebo ručně:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8080 --app-dir .
   ```

Potom otevřete http://localhost:8080 – načte se plná aplikace. Endpoint `/healthz` vrátí JSON `{ "status": "ok", "environment": "<hodnota ALTO_WEB_ENVIRONMENT>" }`.

## Docker image

Dockerfile je připravený tak, aby:

- nainstaloval Python závislosti
- nainstaloval Node.js + npm a zbuildil původní TypeScript (`npm ci && npx tsc`), takže `dist/run_original.js` je k dispozici i v kontejneru
- spustil FastAPI přes `uvicorn`

Build a run (ze složky `web_service/`):

```bash
docker build -t alto-web .
docker run --rm -p 8080:8080 --env-file .env alto-web
```

Environment proměnné pro FastAPI začínají prefixem `ALTO_WEB_` (např. `ALTO_WEB_ENVIRONMENT=production`). `.env` z kořene se načítá přes `python-dotenv`, takže API klíče pro agenty nemusíte psát do obrazu.

### Docker Compose

Pro pohodlný běh na vlastním serveru je připraven `docker-compose.yml`. Ve `web_service/` stačí:

```bash
cp .env.example .env   # doplnit hodnoty
docker compose up --build -d
```

Volumes mapují složky `agents/` a `config/` z hostitele do kontejneru, takže změny v uložených agentech přežijí restart.

## Další kroky

- připravit CI/CD (např. GitHub Actions → Hetzner registry + deploy)
- definovat produkční proces (systemd unit / Hetzner Cloud Apps) a TLS (např. nginx + certbot)
- rozšířit logging/monitoring a případné background joby pro dlouhé běhy

## Hetzner „píseček“ – co bude potřeba

1. **Repo / CI** – mít Git repo (GitHub/Bitbucket) a workflow, který buildne Docker image a pushne ho do registry (Hetzner Container Registry nebo ghcr.io).  
2. **Server** – zvolit Hetzner Cloud VM (např. CX32) a připravit základní balíčky: Docker/Docker Compose, firewall, automatické aktualizace.  
3. **Secrets** – vytvořit `.env` s produkčními klíči na serveru (necommitovat). Hodí se `systemd-tmpfiles` nebo `pass` pro správu.  
4. **Runtime** – buď Docker Compose (poskytnutý soubor) nebo systemd service, které spustí `./web_service/start.sh` s `WEB_CONCURRENCY` nastaveným podle počtu CPU.  
5. **Reverse proxy/TLS** – pokud chcete vlastní doménu, přidejte např. Caddy/nginx před kontejner kvůli HTTPS.  
6. **Monitoring/logy** – nastavit sběr logů (journald + `docker logs`), alerting (Healthchecks/UptimeRobot) a případně Sentry pro Python chyby.
