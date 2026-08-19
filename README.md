# Vaani

A real-time voice agent for Hindi and Hinglish, answering Indian government
scheme-eligibility questions over a streamed pipeline you can interrupt
mid-sentence.

**Live demo:** https://vinayakjeet.github.io/vaani/ (backend on Render's free
tier; the first request after a period of inactivity wakes a cold container,
flagged in the UI, see Runbook).

## Problem

The product is a voice agent. The contribution is a measurement: **where the
latency in a cascaded voice agent goes, and which optimisation buys which
milliseconds**, on two stacks, with variance, including the techniques that
bought nothing and the one that was measured wrong before it was measured
right.

A cascaded pipeline (speech-to-text, then a text-reasoning LLM with tool
calls, then text-to-speech) is the boring, correct architecture for this
task: every stage is independently swappable and testable, which a
speech-to-speech model's collapsed pipeline is not, and this project's
five-stage latency comparison depends on those seams existing (see
DECISIONS.md, "Cascaded pipeline, not a speech-to-speech model"). The naive,
unstreamed version of that pipeline, every stage waiting for the one before
it, is the honest floor everything else is measured against:

```
STT   ~1000ms   whole utterance
LLM     983ms   whole reply
TTS   ~2000ms   243 characters
total  3859ms   end to end
```

That is unusable. The target, per SPEC's own budget, is sub-1000ms time to
first audio measured from the last frame of user speech, not from
endpoint-detected: most published voice-agent latency figures start the
clock at the second point, which quietly removes 250 to 700ms of the wait a
listener actually experiences (see Benchmarks, and DECISIONS.md on the
industry-median citation).

## Architecture

Full diagrams, one turn overlapped, barge-in, and the deployed provider
graph, in [`docs/architecture.md`](docs/architecture.md).

Five stages: VAD/endpoint detection, streaming STT, an LLM turn with tool
calls against fixture eligibility data, a grounding check that refuses any
number the tools never returned, and streaming TTS synthesising sentence by
sentence as the reply arrives. Every stage is Spanlight-instrumented from
the first commit (`vaani/spans.py`'s contract, enforced at the point of
use), and `bench/stages.md` defines exactly what each stage span starts and
ends at, hashed before anything was measured against it.

## Benchmarks

All numbers below are reproducible: `uv run python -m bench.<script>`,
source in `bench/`. Dated against `openai/gpt-oss-120b`, not the model
originally planned; Groq removed `llama-3.3-70b-versatile` from its catalog
mid-project (see What Broke).

**Per-stage waterfall**, twenty corpus utterances, one pass, free stack (Groq
+ EdgeTts), `bench/waterfall.py`:

```
stage            n      p50      p95
vad.endpoint    20   5418.8   6042.9
stt.stream      20  10966.1  14634.1
llm.generate    20   5203.8  11011.5
tts.synthesize  20   3139.2   7925.8
turn            20   6195.3   6804.9

headline                    n      p50      p95
first_audio_ms             20   1464.9   1516.6
first_answer_audio_ms      19  14332.3  18310.0
first_answer_heard_ms      19  14332.4  18310.1
```

`first_answer_heard_ms` is the number judged against the sub-1000ms target,
per `bench/stages.md`; the other two are published beside it because each
flatters in a specific, named way (filler-covered turns, and send time
rather than playout time). This run's own filler rate was 1.0: every turn
needed the acknowledgement, which is real queue and reasoning overhead on
the current model, not a synthesised result, and is the single clearest
argument for M5's still-open ablation.

**Tail multiplication**, checked against the same real trace,
`bench/tail_multiplication.py`: 25.0% of turns landed at or past at least
one stage's own p95, against 22.6% predicted by `1 - 0.95^5` if five serial
stages fail independently five percent of the time each. Close to the
theoretical figure at a sample size honest enough to state the caveat: at
n=20 a stage's own p95 is one below its worst observed run, so this is
really asking whether stages' worst runs land on the same turn, not testing
an unusual few percent of a large sample.

**Endpointing frontier**, latency bought against how many of a battery of
plausible mid-utterance pauses each setting would cut off,
`bench/endpointing_frontier.py`:

```
aggressiveness  trailing_ms  shortest cut (ms)  cut/tested
0                      1000               1000       3/12
1                       840                900       4/12
2                       700                700       6/12
3                       500                500       8/12
```

No real false-endpoint rate exists to pair with this: that needs recorded
speech with genuine disfluency, which this project does not have, stated as
a limitation rather than filled with a guess.

**Prompt caching**, `bench/prompt_cache.py`: measured, and the finding is a
real null result. Caching activates on this project's own system prompt and
tool schemas (`cached_tokens` read 256 on 6 of 10 identical-prefix calls),
but the two arms' time-to-first-token did not separate (557.4ms median with
caching eligible against 473.1ms without), both swamped by ordinary queue
and network variance far larger than 256 cached tokens could plausibly save.
Not a technique this project uses or reports as a win.

**Depth-chapter ablation**, twenty utterances per arm, interleaved,
`bench/ablation.py`: three real independently-switchable arms (of the seven
originally planned; the other four were never built or share one
implementation, see `ablation/hypothesis.md`).

```
arm            n  median_ms  p95_ms  min_ms   max_ms  bought_vs_unstreamed
unstreamed    20     5666.7  8253.1  3402.5   9632.1
streamed      20    11114.3 14751.4  5716.0  14992.2            -5447.6ms
semantic_off  20    10793.4 14158.6  4984.5  16490.4            -5126.6ms
```

**Streaming loses to the naive baseline**, every one of 20 utterances, and
this contradicts what `ablation/hypothesis.md` predicted before any of this
was measured. It is not noise or a clock bug: two real measurement bugs were
found and fixed on the way here (see What Broke), and this result held
across the clean re-runs after both fixes. The mechanism is traceable in the
code: `TurnClock.mark_heard` records the answer as heard only once whatever
audio is already queued ahead of it has played, which on nearly every real
turn is the M3.5 filler acknowledgement (filler rate 1.0 in the waterfall
run above). The unstreamed baseline has no filler at all, so it pays no such
tax; the filler plays in full regardless of how much of the real answer is
already ready by the time it finishes, so it never shortens the wait it
exists to cover and sometimes lengthens it. That is a genuine finding about
M3.5's filler design, not about streaming, and the natural fix (make the
filler interruptible by the real answer, then re-measure) is BACKLOG's
M5.2c. Full account, including how the two prior measurement bugs were
caught, in DECISIONS.md.

**Eval set**, 50 scripted scenarios across 10 categories, `eval/build_scenarios.py`
and `eval/run_eval.py`: exact structural matching against what the pipeline
actually decided (tool called, scheme id, eligibility verdict), never a
free-text judge scoring reply prose, so no judge calibration applies here
(BACKLOG's M4.8). Stable at **80% (40/50)** across three independent live
runs, after fixing two real bugs the first run itself surfaced (a required
income field that should not have been required, a runner that silently
dropped rejected tool calls from its own record) and correcting 13 of the
eval's own labels, which is why the very first run scored 48%. Every
remaining failure falls into four already-understood categories, none a
schema or label defect: the model sometimes does not finish the tool chain it
started, sometimes sends a scheme's display name instead of its id, and the
fixture `find_schemes`'s keyword matching both misses a real scheme on
unusual phrasing and, once, matched a disability-pension question to the
old-age-pension scheme on the shared word "pension." **Not a CI gate yet**:
the labels need a human review pass before 80% means anything gate-worthy,
per this file's own header and Spanlight's 35.8%-mislabeled-corpus lesson.

**Threats to validity**: [`ablation/threats-to-validity.md`](ablation/threats-to-validity.md),
led by the one this session actually hit rather than a textbook list, sustained
account throttling from this session's own testing volume, discovered mid-way
through closing M5.2c. The published ablation table above predates that fix;
stated there in the same place, not left implicit.

## Technical decisions

Full log with context, alternatives, and consequences: [`DECISIONS.md`](DECISIONS.md),
41 entries. Highlights:

- **WebSocket, not WebRTC**, for the transport: this project's audio never
  crosses a lossy public network path the way a real call does, and WebRTC's
  signalling and TURN infrastructure would have spent time on plumbing
  orthogonal to what this project measures.
- **Cascaded, not speech-to-speech**: the whole depth chapter depends on
  independently swappable, independently measurable stages an S2S model
  does not have.
- **Semantic endpointing** cuts the trailing-silence wait to roughly 200ms
  when a partial transcript already looks grammatically finished, the
  largest single term in the optimised latency budget, at a false-endpoint
  cost this project states it cannot yet measure honestly.
- **A hedged second LLM request**, not a serial failover: firing a second
  request at the measured p90 rather than failing over serially, because a
  serial failover pays the tail twice. The threshold itself was measured,
  not guessed, and the first shipped value was wrong by a wide enough margin
  to matter (see What Broke).

## What broke

Recorded here because a README that only shows the finished thing is
marketing, not an engineering record.

- **Groq removed the model this whole project was built against, live, mid-session.**
  `llama-3.3-70b-versatile` returned 404 `model_not_found`; every real turn on
  the deployed service was broken until the model was swapped to
  `openai/gpt-oss-120b`, which turned out to be a reasoning model that could
  spend its entire token budget on hidden reasoning tokens and return zero
  content unless `reasoning_effort` was explicitly set low. Fixed and a
  general guard added for the same failure shape from any future model.
- **A generator abandoned mid-span corrupts the tracer for every turn after it.**
  Several places in the pipeline stopped reading a streaming generator before
  it reached its own natural end (an interruption, an early `break`, a bench
  script cancelling a slow turn); the generator's OpenTelemetry span was then
  closed by the garbage collector, in the wrong task, and the resulting
  detach failure was silent. Twenty turns in a row was enough to stop
  unrelated spans from recording at all. Every such call site now closes its
  generator explicitly, in the task that opened it.
- **A disconnect mid-reply was recorded as a reply delivered in full.** The
  session decided whether to commit a reply to its own conversation history
  by checking whether playback state was still "speaking," which is correct
  for an interruption and wrong for a disconnect, since a disconnect cancels
  playback without moving that state first. A turn cut off after one chunk
  of four was logged as the whole answer, heard.
- **Sarvam's streaming STT is not a request-response protocol.** The first
  integration assumed one transcript per audio chunk sent and hung
  indefinitely against the real socket: the server runs its own VAD and
  emits transcripts on its own schedule. Rebuilt around a concurrent sender
  and response pump.
- **An ablation baseline's own clock skipped the wait it was supposed to
  pay**, making the slow, naive baseline measure faster than the optimised
  path. Caught before publishing anything from it.
- **`check_eligibility` was almost never actually being called, on the live
  deployed service.** The tool schema declared its nested `applicant` object
  as a `$ref` into `$defs`, which is exactly correct JSON Schema and exactly
  what pydantic emits for a nested model. Groq's `openai/gpt-oss-120b`
  cannot reliably resolve it: shown the schema with the `$ref` in place, it
  invented its own field names for `applicant` 5 of 5 tries against the live
  endpoint; shown the identical schema with the reference inlined, it got
  every field right 5 of 5 tries. Every real eligibility question the
  deployed service answered since the model swap below was very likely a
  "could not check" filler line, not a real check. Fixed by inlining every
  `$ref` before a schema is advertised, deployed, and verified live.
- **A latency ablation's own naive baseline turned out to be the fair one.**
  The optimised, streamed arm measured slower than the unstreamed baseline
  at a clean n=20, and the reason was not the clock this time: the filler
  acknowledgement plays in full before the real answer regardless of how
  much of the answer is already ready, a real design gap the ablation
  surfaced rather than a measurement mistake. See Benchmarks above.

## Runbook

```
uv sync
uv run uvicorn app.main:app --reload
```

Serve `web/` and open it; needs `GROQ_API_KEY` in `.env`. Cold start on the
deployed backend (Render free tier, spins down when idle) is flagged in the
client UI rather than left to read as a hang.

Reproduce any number above: `uv run python -m bench.<script>` (`waterfall`,
`tail_multiplication`, `endpointing_frontier`, `prompt_cache`, `ablation`),
each writing its own raw per-run JSON alongside the printed summary. The
exact raw data behind every number actually published here is committed in
[`bench/results/`](bench/results/) and [`eval/results/`](eval/results/),
so checking a figure never requires spending a live API call.
