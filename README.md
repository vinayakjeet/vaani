# Vaani

A real-time voice agent for Hindi and Hinglish, answering government
scheme-eligibility questions over a streamed pipeline you can interrupt
mid-sentence.

**In progress.** Project three of eleven. The parts that work today are the
walking skeleton: browser microphone to a deployed server and back as spoken
audio, with per-stage tracing.

## What it is actually for

The product is a voice agent. The contribution is a measurement: **where the
latency in a cascaded voice agent goes, and which optimisation buys which
milliseconds**, on two stacks, with variance, including the techniques that
bought nothing.

The first honest number, from the naive pipeline where every stage waits for the
one before it:

```
STT   ~1000ms   whole utterance
LLM     983ms   whole reply
TTS   ~2000ms   243 characters
total  3859ms   end to end
```

That is unusable, and it is the baseline on purpose. The goal is sub-1000ms time
to first audio, measured from the last frame of user speech rather than from
endpoint-detected, because that is the wait a person actually experiences.

## Running it

```
uv sync
uv run uvicorn app.main:app --reload
```

Then serve `web/` and open it. Needs `GROQ_API_KEY` in `.env`.

## Not finished

Benchmarks, the eval set, and the ablation are not built yet. No number here is
final and none of them are in a README section that claims to be.
