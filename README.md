# ai-portfolio-template

Foundation template for a portfolio of production-AI projects. Fork this repo per
project; keep the six sections below, replace the placeholders with the specifics
of what you built.

## Problem
_TODO: what problem does this project solve, for whom, and why does it matter?_

## Architecture
_TODO: diagram or description of the system - services, data flow, external
providers used from `llm/providers/quotas.yaml`._

## Benchmarks
_TODO: latency/cost/accuracy numbers that back up any claims this project makes._

## Technical Decisions
_TODO: link to the relevant [DECISIONS.md](DECISIONS.md) entries; don't duplicate
their content here._

## What Broke
_TODO: real incidents/bugs hit while building this, and how they were fixed or
worked around. See [LEARNING.md](LEARNING.md) for the running log._

## Run It

Requires [uv](https://docs.astral.sh/uv/) and Docker.

**Local, no Docker:**
```
uv sync
uv run uvicorn app.main:app --reload
```

**Docker Compose** (app + postgres/pgvector + redis, no API keys required -
defaults to the mock LLM provider):
```
docker compose up --build
```
Then:
```
curl localhost:8000/healthz
curl localhost:8000/version
curl -X POST localhost:8000/demo/chat -H "Content-Type: application/json" -d "{\"prompt\": \"hello\"}"
```

**Tests / lint:**
```
uv run pytest
uv run ruff check .
```

**Using a real LLM provider:** copy `.env.example` to `.env`, set the relevant
`*_API_KEY` and `LLM_PROVIDER`, see [QUOTAS.md](QUOTAS.md) for free-tier limits
per provider.
