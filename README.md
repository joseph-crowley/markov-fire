# Fat Tailed Solutions — Markov Fire Portal

A full-stack Django application from Fat Tailed Solutions for configuring, running, and visualising stochastic wildfire simulations that blend temporal Markov processes with spatial fire-spread dynamics. The portal provides:

- Rich configuration management for simulation presets and parameter tuning
- Background execution via Celery with persisted tick-by-tick results in PostgreSQL
- Real-time visualisation of the fire grid and key metrics using Django Channels and WebSockets
- REST API endpoints for external automation and integrations
- Docker Compose stack with Django (ASGI), Celery worker/beat, Redis, and PostgreSQL

## Features

- **Configurable Engines**: Temporal spread, extinguish, and suppression rates combined with spatial grid dynamics, environmental modifiers, and firefighting resources.
- **Live Streaming**: WebSocket feed broadcasts every simulation tick to all connected clients; dashboard renders heatmap grid and charts in real time.
- **Persistence**: Simulation inputs, runs, and individual ticks stored for replay and analytics.
- **Background Tasks**: Celery workers execute simulations asynchronously; progress and completion events push to clients immediately.
- **REST API**: Create configurations, schedule runs, and fetch tick history programmatically.

## Quick Start (Docker)

```bash
docker compose up --build
```

Set `APP_PORT` in `.env` (defaults to `8000`) to control the exposed port. The provided `.env` sets it to `8001`; visit `http://localhost:8001/` (or your chosen port) and log in with a Django superuser (create one via `docker compose run --rm web python manage.py createsuperuser`).

From the homepage you can:

- Explore the Fat Tailed Solutions overview describing the project mission and workflow.
- Jump straight to “Launch a Simulation” or “Browse Configurations” via CTA buttons.
- Access navigation links for configurations, active runs, and new run creation.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DJANGO_DB_ENGINE=django.db.backends.sqlite3
python manage.py migrate
python manage.py runserver
```

Optional: start Celery and Redis for live streaming locally.

```bash
redis-server
celery -A markov_fire_portal worker --loglevel=info
```

## Architecture Overview

- `simulation/` – domain models, Celery tasks, REST API, and the wildfire simulation engine (`services/engine.py`).
- `control/` – Django views/templates for configuration and monitoring dashboards.
- `streaming/` – Django Channels consumer that multiplexes simulation ticks to WebSocket clients.
- `markov_fire_portal/` – project settings, URLs, ASGI/WSGI entrypoints, Celery integration.

## REST Endpoints

- `POST /api/configs/` – create new configuration (authenticated).
- `POST /api/runs/` – schedule a run for a configuration (authenticated).
- `GET /api/runs/{id}/ticks?from=0&limit=200` – retrieve stored tick metrics.

## WebSocket Protocol

Connect to `ws://<host>/ws/simulations/<run_id>/`.

Messages:
- `{"type": "tick", ...}` – every new tick with grid state and metrics.
- `{"type": "completed", ...}` – run finished.
- `{"type": "failed", "error": "..."}` – run aborted.
- Send `{"type": "catchup", "from": 0}` to request persisted history.

## Testing

```bash
pytest
```

(or `python manage.py test` after installing `pytest-django` or using Django's test runner).

## Roadmap

- Geospatial overlays (wind fields, terrain data)
- Advanced resource strategies and multi-run analytics
- Export & replay tools for historical simulations

## License

MIT
