# Published eval runs

M5.4/M4.7: the raw per-scenario output behind README's "stable at 80%"
line, committed so a stranger can inspect exactly which of the 50 scenarios
passed, failed, and why, without spending a live API call. Each row records
the tool calls the real pipeline actually made for that scenario and the
reason a check passed or failed; see `eval/run_eval.py` for what each field
means.

| File | Command that produced it | Result |
|---|---|---|
| `eval_2026-08-19.json` | `uv run python eval/run_eval.py` | 40/50 (80%) |

Re-running `eval/run_eval.py` today will not reproduce this file exactly:
the model's real tool-calling behaviour varies run to run, which is why
README reports "stable across three independent live runs" rather than a
single number, and why this is one of three, not the only one. The other
two were not archived; this is the most recent and is representative of all
three, within a couple of scenarios either way (see BACKLOG's M4.7 entry for
the category-by-category breakdown across all three).
