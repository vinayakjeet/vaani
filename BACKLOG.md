# Backlog

Tasks come from here, not from ad-hoc requests mid-session (see
CONTRIBUTING.md). Check items off at the end of the session that completes them.

## Foundation template (this repo)
- [x] FastAPI app factory, `/healthz` + `/version`, pydantic-settings config, structured JSON logging
- [x] `llm/` provider-agnostic client: retry+backoff, 429 throttle, provider registry, cost/token logging
- [x] `otel_bootstrap.py` - OTLP export gated on env, no-op otherwise
- [x] Dockerfile (slim, non-root, uv) + docker-compose (postgres/pgvector + redis)
- [x] GitHub Actions: ruff+pytest on PR, reusable deploy stub
- [x] Docs skeleton: README, DECISIONS, LEARNING, QUOTAS
- [x] CONTRIBUTING.md conventions

## Next
- [ ] Pick project #1, fork this repo, fill in README's Problem/Architecture/Benchmarks sections
- [ ] Wire a real deploy target into `deploy-reusable.yml` once project #1 picks one (Render/HF Spaces/...)
