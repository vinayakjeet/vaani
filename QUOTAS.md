# LLM Provider Quotas

Human-readable mirror of [`llm/providers/quotas.yaml`](llm/providers/quotas.yaml)
(the machine-readable config `llm/providers/registry.py` actually loads). The
two are **not auto-synced** - update both when a limit, price, or model changes,
and record why in [DECISIONS.md](DECISIONS.md) if it's a meaningful shift.

Measured against real keys on 2026-08-06 while building ShipGate, not copied
from documentation. Four findings that each cost real debugging time:

- **Gemini sends no `Retry-After` header.** The delay appears only in prose
  inside the JSON error body, and separately in a `google.rpc.RetryInfo`
  detail. A client reading the header alone falls back to a short default and
  hammers a limit needing most of a minute.
- **Gemini reports sub-second waits in milliseconds.** "Please retry in
  607.269104ms" parsed as seconds stalls a run for ten minutes over a delay of
  six tenths of a second.
- **Gemini bills reasoning tokens that appear in neither itemised usage
  field.** A real response read prompt 2, completion 9, total 197. Reading the
  itemised fields undercounts by roughly eighteen times, which is why the
  gemini entry uses `usage_parser: total_aware`.
- **Groq's binding limit on the 70B model is tokens per minute, not
  requests.** Prompts of roughly 400 tokens exhaust it long before the 30
  requests/min ceiling, and the resulting wait measured 350 seconds. A tiny
  request still succeeds during that window, which makes it easy to
  misdiagnose as the service working normally. `llama-3.1-8b-instant` has real
  headroom.

Also: `gemini-2.0-flash` returns a quota error on a new key and
`gemini-2.5-flash` is retired for new users. Use `gemini-flash-latest`, which
is an alias and will move, so record the model on every run.

Free-tier limits change without notice. "Last verified" is when a human last
checked the provider's own docs/dashboard - if it's more than a month old,
re-verify before relying on it for capacity planning.

| Provider | Free-tier limit | Reset window | Last verified |
|---|---|---|---|
| gemini | 20 requests/window, `gemini-flash-latest` | rolling, 20 to 49s recovery | 2026-08-06 |
| groq | 30 requests/min, but tokens/min binds first on 70B | per minute | 2026-08-06 |
| cerebras | 30 requests/min | per minute | 2026-08-04 |
| openrouter | 20 requests/min (`:free` model suffix) | per minute | 2026-08-04 |
| ollama | none (self-hosted/local) | n/a | 2026-08-04 |
| sarvam | 10 requests/min | per minute | 2026-08-04 |
| mock | none (in-process, no network) | n/a | n/a |

Verify against each provider's own current docs/dashboard before depending on
these numbers for capacity planning - this table is a starting point, not a
guarantee, and exact rate-limit doc URLs move around often enough that they're
deliberately not pasted here. Search "<provider> API rate limits" for each.
