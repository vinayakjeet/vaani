# Published runs

M5.4: the raw data behind every number in the top-level README's Benchmarks
section and BACKLOG's M5 entries, committed rather than left to only exist on
whoever's machine ran the script. A stranger can recompute every published
figure from these files without spending a single live API call; regenerating
a *new* run with `uv run python -m bench.<script>` will not reproduce the same
numbers exactly, since the underlying model's real latency and, on the
eligibility rows, its real answers vary run to run. That variance is the
subject, not a defect in this archive.

| File | Backs | Command that produced it |
|---|---|---|
| `waterfall_2026-08-17.json` | README's per-stage waterfall table | `uv run python -m bench.waterfall --n 20` |
| `ablation_n20_2026-08-18.json` | README's and BACKLOG M5.2's ablation table | `uv run python -m bench.ablation --n 20` |
| `prompt_cache_2026-08-17.json` | README's prompt-caching null result | `uv run python -m bench.prompt_cache` |

Not archived here: `bench/tail_multiplication.py`'s output, which is derived
entirely from `waterfall_2026-08-17.json`'s own raw per-stage data and needs
no separate live run to recompute; and `bench/endpointing_frontier.py`'s
output, which sweeps a real `Endpointer` against synthetic pause durations
and carries no LLM or network call at all, so re-running it reproduces the
exact same table every time.

`bench/bargein.py`'s own barge-in latency numbers predate this archive and
are not yet re-collected here; BACKLOG's M2.4 entry carries the run that
produced the published figures.
