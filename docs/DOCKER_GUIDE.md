# ChurnOps — Docker Guide

> How and why we containerize the ChurnOps application.

---

## Why Docker?

| Problem | Docker Solution |
|---------|-----------------|
| "Works on my machine" — different Python versions, OS differences | Container includes the exact Python version, OS, and all dependencies |
| Dependency hell — conflicting library versions | Each container is isolated with its own dependency tree |
| Deployment complexity — setting up servers manually | `docker-compose up` starts everything in one command |
| Scaling — need multiple API instances | Containers can be orchestrated with Kubernetes or Docker Swarm |
| Reproducibility — builds must be identical every time | Docker images are immutable — same build = same result |

---

## Architecture

```
┌──────────────────────────────────┐
│       docker-compose.yml         │
│                                  │
│  ┌────────────┐  ┌────────────┐  │
│  │  churnops-  │  │  churnops-  │ │
│  │    api      │  │  frontend   │ │
│  │  :8000      │  │    :3000    │ │
│  │             │  │             │ │
│  │  FastAPI +  │  │  Nginx +    │ │
│  │  Uvicorn    │  │  React      │ │
│  └──────┬──────┘  └──────┬──────┘ │
│         │                │        │
│    Dockerfile       Dockerfile    │
│    (root)          (frontend/)    │
└──────────────────────────────────┘
```

---

## Files Explained

### `Dockerfile` (Backend — Project Root)

```dockerfile
FROM python:3.10-slim AS base       # Minimal Python image (~150MB vs ~900MB full)
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y build-essential

# Python deps (separate layer for caching)
COPY requirements-api.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml .
RUN pip install -e .

EXPOSE 8000

# Health check for orchestrators
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key decisions:**
- **`python:3.10-slim`** — Slim variant is much smaller than the full image, reducing build time and attack surface
- **`requirements-api.txt`** — A lighter dependency file (no training deps like Optuna, Great Expectations) for a smaller production image
- **Layer caching** — Dependencies are installed before copying code. This means code changes don't trigger a full reinstall
- **`HEALTHCHECK`** — Docker/Kubernetes uses this to know when the container is ready to serve traffic

### `frontend/Dockerfile` (Frontend)

Builds the React app and serves it via Nginx on port 80.

### `docker-compose.yml`

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    healthcheck: ...

  frontend:
    build: ./frontend
    ports: ["3000:80"]
    depends_on:
      api:
        condition: service_healthy
```

**Key decisions:**
- **`depends_on` with `service_healthy`** — Frontend won't start until the API passes its health check. This prevents the UI from loading before the backend is ready
- **Port mapping** — API on 8000, Frontend on 3000 (standard convention)

---

## Usage

### Start Everything
```bash
docker-compose up --build
```
This builds both images and starts them. The `--build` flag ensures you get the latest code.

### Start in Background
```bash
docker-compose up -d --build
```

### View Logs
```bash
docker-compose logs -f api        # API logs only
docker-compose logs -f frontend   # Frontend logs only
docker-compose logs -f            # All logs
```

### Stop Everything
```bash
docker-compose down
```

### Rebuild After Code Changes
```bash
docker-compose up --build
```

---

## Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `PYTHONUNBUFFERED` | api | `1` | Ensures Python logs appear in real-time |
| `VITE_API_URL` | frontend | `http://api:8000` | Backend URL (uses Docker DNS) |

---

## Production Considerations

### 1. Use Multi-Stage Builds
The backend Dockerfile could be split into a build stage and a runtime stage to reduce the final image size.

### 2. Add a Reverse Proxy
In production, put Nginx or Traefik in front of both services for:
- SSL termination (HTTPS)
- Rate limiting
- Load balancing

### 3. Model Artifacts
In production, models should be loaded from a remote model registry (MLflow on a server, or S3) rather than local files. The current setup loads from the local `mlruns/` directory.

### 4. Secrets Management
Use Docker secrets or environment variable injection (not hardcoded) for:
- Database credentials
- API keys
- MLflow tracking URIs
