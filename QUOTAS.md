# LLM Provider Quotas

Human-readable mirror of [`llm/providers/quotas.yaml`](llm/providers/quotas.yaml)
(the machine-readable config `llm/providers/registry.py` actually loads). The
two are **not auto-synced** - update both when a limit, price, or model changes,
and record why in [DECISIONS.md](DECISIONS.md) if it's a meaningful shift.

Free-tier limits change without notice. "Last verified" is when a human last
checked the provider's own docs/dashboard - if it's more than a month old,
re-verify before relying on it for capacity planning.

| Provider | Free-tier limit | Reset window | Last verified |
|---|---|---|---|
| gemini | 15 requests/min (gemini-2.0-flash) | per minute | 2026-08-04 |
| groq | 30 requests/min | per minute | 2026-08-04 |
| cerebras | 30 requests/min | per minute | 2026-08-04 |
| openrouter | 20 requests/min (`:free` model suffix) | per minute | 2026-08-04 |
| ollama | none (self-hosted/local) | n/a | 2026-08-04 |
| sarvam | 10 requests/min | per minute | 2026-08-04 |
| mock | none (in-process, no network) | n/a | n/a |

Verify against each provider's own current docs/dashboard before depending on
these numbers for capacity planning - this table is a starting point, not a
guarantee, and exact rate-limit doc URLs move around often enough that they're
deliberately not pasted here. Search "<provider> API rate limits" for each.
