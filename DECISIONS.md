# Decisions

Every nontrivial choice gets an entry here at the time it's made - not
reconstructed later from memory. Newest entries at the top.

## Format
```
## YYYY-MM-DD: <short title>
**Context:** what problem/question forced a decision.
**Decision:** what was chosen.
**Alternatives considered:** what else was on the table, and why it lost.
**Consequences:** what this makes easier/harder later.
```

## 2026-08-18: check_eligibility was never actually being called, and the eval that was supposed to catch it is what caught it

**Context:** M4.7's eval runner (`eval/run_eval.py`), wrapping `vaani.llm_turn.dispatch`
to record real tool calls against `eval/scenarios.json`, was run live for the first time
as a smoke test. Result: 0 of 12 `eligibility_positive` scenarios passed, and all 3
`boundary` scenarios failed the same way. `find_schemes` was called correctly every time;
`check_eligibility` either was never called at all, or Groq rejected it outright with
`tool_use_failed`, quoting a `failed_generation` with invented field names for `applicant`
(`land_acres`, `land_area_acres`, `annual_income`, `age`, `income`, `occupation`,
`family_members` - never the four real ones). This was live against the currently deployed
model, meaning the deployed service was silently failing essentially every eligibility
question and falling back to `COULD_NOT_CHECK`, not a scenario-corpus artifact.

Isolated with a direct probe against the real Groq endpoint, two schemas, five tries each,
same question, same conversation state (`find_schemes` already answered, feeding a real
tool result back): the exact schema `tool_schemas()` was already sending, which declares
`applicant` as `{"$ref": "#/$defs/Applicant"}`, failed 5 of 5 tries with invented field
names. The same schema with `Applicant`'s definition inlined in place of the `$ref`
succeeded 5 of 5 tries, with the exact declared field names and correct values pulled from
the conversation (`land_holding_acres: 3`, `state: "Bihar"`, `annual_income_inr: 0` as the
correct default when income was never mentioned). `find_schemes` has no nested object and
no `$ref` in its own schema, which is why it was never affected.

This is the same failure class the `Rupees`/`Acres` type aliases in `vaani/tools.py`
were already built to avoid, one layer up: "a reference it may not resolve is a worse
contract than a repeated four-line object." That fix covered field-level aliases; it never
reached `Applicant`, the one nested `BaseModel`, because pydantic's own `model_json_schema()`
always renders a nested `BaseModel` as `$ref`/`$defs` and nothing in the code touched that
shape afterward.

**Decision:** added `_inline_defs()` in `vaani/tools.py`, run over every tool's schema
before it is advertised: walks the schema, replaces every `$ref` into `$defs` with the
definition it points to, and drops `$defs` from the output. `tool_schemas()` now never
emits a `$ref`, for `Applicant` or for any future nested tool argument. A regression test
(`test_no_schema_ever_advertises_a_ref`) walks every advertised schema and asserts neither
key ever appears again. Re-ran the `eligibility_positive` eval category after the fix:
9 of 12 passed, up from 0 of 12; the one remaining failure is the model choosing not to
call `check_eligibility` on a fully-answerable Devanagari-script question, a different and
much smaller failure mode (an omission, not a malformed call) that this entry does not
claim to have fixed.

**Alternatives considered:** flattening `Applicant`'s fields directly onto
`EligibilityRequest` (dropping the nested model entirely), rejected because it would also
change the internal call shape `check_eligibility()` and its tests build against, for a
problem that is specifically about what the model is shown, not what the code is shaped
like internally. A model-level fix (switching providers, or accepting the loss and coaching
the model harder in the system prompt) was not tried first, on the same reasoning as the
`gpt-oss-120b` swap earlier this project: fix the contract before assuming the model cannot
be told correctly, since here it plainly could be.

**Consequences:** every real eligibility answer the deployed service has given since the
model swap to `openai/gpt-oss-120b` (see the entry below on Groq removing
`llama-3.3-70b-versatile`) was very likely a `COULD_NOT_CHECK` filler line dressed as an
answer, not a real check, for as long as that model has been live and this schema shape
unchanged. Deployed immediately after verification rather than batched with other work,
since this is a live correctness bug, not a benchmark finding. The n=20 ablation run
started before this fix was discovered was discarded rather than published: an unknown
share of its `check_eligibility`-bearing turns paid for a failed second round before
falling back, on both arms, in a way this entry has no basis to claim was symmetric.

## 2026-08-18: Sarvam Saaras is not a request-response protocol, and the first build assumed it was

**Context:** M4.4's second stack needed a streaming STT client for Sarvam Saaras. The
public documentation for the streaming WebSocket endpoint was inconsistent across pages
at the time this was written (two reference pages 404ed, a third gave a partial picture),
so the vendor's own Python SDK (`sarvamai`) was added as a dev dependency rather than
hand-rolling the wire format against a 100-credit balance with no way to iterate cheaply.

The first implementation sent one message via `transcribe()` and awaited exactly one
`recv()` per send, mirroring `ChunkedStt`'s own request-response shape. It hung
indefinitely against the real socket: `connect()` and the first `transcribe()` both
succeeded fast, and `recv()` then waited without returning anything, for as long as the
call was allowed to run. QUOTAS.md's own note, written before this session, already said
why: "Saaras streaming has its own VAD with adjustable sensitivity." The server decides on
its own schedule when a transcript is worth emitting; it is not obligated to answer each
message with exactly one response, and nothing in the SDK's exposed interface pairs a send
to a specific reply.

**Decision:** rebuilt around two concurrent tasks sharing one connection: a sender that
pushes audio without waiting on anything, and a pump that reads every response the server
produces, in arrival order, onto a queue `stream()` reads from and yields as partials.
`RESPONSE_GRACE_S` (8s) bounds how long the read side waits for one more response once
sending has finished, and it resets on every response actually received, so it is a
silence bound rather than a guess at how long a real reply takes.

A second bug surfaced in the same investigation, independent of the first: audio shorter
than one send interval never reached the server before `flush()` asked it to finalize,
because the interval-based sender only sends when the threshold is crossed and never had
a catch-up step for whatever was still buffered when the frames ran out. That is most
backchannels, "haan", "theek hai", and any utterance under 600ms. Fixed by always sending
the remainder, even under threshold, immediately before flushing.

**Alternatives considered:** hand-rolling the WebSocket protocol directly rather than
depending on the vendor's SDK, rejected for the reason the SDK was chosen in the first
place: the framing is not fully, consistently documented publicly, and confirming it by
trial against a small, non-renewable credit balance is a worse bet than trusting the
vendor's own maintained client, even though that client's own request-shaped `transcribe`
helper turned out to invite exactly the wrong mental model here. Guessing a shorter, fixed
total timeout instead of a resetting grace window, rejected because a fixed bound cannot
tell a slow-but-still-arriving stream of responses from a genuinely finished one without
either cutting a real reply short or waiting the full bound on every call regardless of
how quickly it actually finished.

**Consequences:** `tests/vaani/test_sarvam.py` scripts the server as something that pushes
responses on its own schedule, not as something that answers each send, which is now the
correct mental model for anyone extending this file. Verified for real exactly once, end
to end, against `bench/corpus/scheme-pm-kisan.wav`: the returned transcript matched the
corpus text's meaning exactly. The full twenty-utterance comparison run this milestone
still needs is not done, deliberately: it spends real credits with no programmatic way to
read the balance back, and that decision belongs to whoever can read the dashboard, not to
a script running unattended.

## 2026-08-17: hedge_after_ms was below every measured sample, so the hedge fired on nearly every call

**Context:** M3.5's own note said `hedge_after_ms` "belongs at the primary's measured p90
and nothing has been measured, so it moves when M0.1 exists." M0.1 is still blocked on
manual dashboard checks, but `bench/prompt_cache.py`'s own twenty real time-to-first-token
samples against `openai/gpt-oss-120b`, gathered for a different measurement earlier
tonight, are a real distribution for the same primary provider. p50 499.7ms, p90 833.9ms.
The shipped default was 300ms, below every one of the twenty samples: `sorted(...)` put
the minimum at 411ms. A hedge meant to fire on roughly a tenth of calls was firing on all
of them, doubling real provider spend on ordinary traffic rather than only covering the
tail.

**Decision:** `DEFAULT_HEDGE_AFTER_MS = 834`, the measured p90, with the measurement and
its date in the comment beside it rather than only in this entry.

**Alternatives considered:** waiting for M0.1's own dashboard-derived numbers before
touching this, rejected because the twenty samples already in hand are a real measurement
of the same thing against the same provider, not a guess standing in for one, and shipping
a known-wrong default until an unrelated blocked task clears is worse than updating it now
and revising again if M0.1's own numbers disagree.

**Consequences:** hedging now costs a second request on roughly the intended tenth of
calls rather than on all of them. `tests/fault/test_llm_timeout.py` is unrelated to this
specific number but closes M3.5's other open half: a hung provider, injected for real via
`fault.faulty_endpoint(Fault.HANG)`, ends the turn in a bounded time rather than the
session waiting out the hang, verified at both the point before the model has said
anything and the point where a tool round is already in flight.

## 2026-08-17: No rate limiting beyond the single-session lock

**Context:** M3.8's own title names "rate limiting and refusal beyond one session." The
refusal half already worked and is now tested (`tests/app/test_concurrency_refusal.py`);
rate limiting the refusal path itself was the remaining open question.

**Decision:** not built. The single-session lock already bounds concurrent backend work,
every provider call and every compute path, to exactly one session at a time, and a
refused connection costs a socket accept and one JSON message, nothing a rate limit would
meaningfully protect. Adding one anyway means picking a number, requests per minute per
IP or similar, with no observed abuse pattern to size it against.

**Alternatives considered:** a small, provisional per-IP limit on new connection attempts,
rejected on the record this project has already made twice this session: ShipGate gated on
2 points against a measured noise floor of 20, Spanlight shipped a threshold that fired on
a pattern written down as healthy. A number invented under no pressure to be right is the
same mistake with a different label.

**Consequences:** if the refusal path itself is ever observed to be abused, meaning
something a rate limit would actually stop, that observation is the number to build
against, and this decision is the one to revisit.

## 2026-08-17: Prompt caching is a measured null result, not a technique this project uses

**Context:** M1.14 asked to measure Groq's automatic prompt caching, on the assumption
stated in its own task text that it cuts latency on the cached prefix. `bench/
prompt_cache.py` sent the real system prompt and real tool schemas, roughly 350 to 390
tokens, in two arms: `repeated`, the identical prefix every call, and `broken`, a fresh
UUID appended so no call's prefix can match a previous one. `usage.prompt_tokens_details.
cached_tokens` read 256 on 6 of 10 `repeated` calls, so caching genuinely activates on
this prompt's real size, and this project's prefix clears `openai/gpt-oss-120b`'s minimum,
whatever it is inside Groq's stated 128-to-1024-token range for the family.

**Decision:** report the null result rather than the win the task assumed. Time to first
token did not separate between the two arms: `repeated` measured slower on the median
(557.4ms against 473.1ms), and both arms' own spread, roughly 400 to 1540ms, is far wider
than any gap between them. 256 cached tokens is a small fraction of a request whose latency
is dominated by Groq's queue time and the network round trip from India to wherever Groq
answers from, and whatever the cache saves on prefix computation does not surface above
that. M5's ablation will not carry a caching row, because there is nothing here to
attribute milliseconds to.

**Alternatives considered:** a larger `--repeats` to shrink the confidence interval and
possibly find a real separation, rejected for now: the two arms' medians are on the wrong
side of each other, not close together, so more samples would narrow noise around a null
finding rather than reveal a hidden effect, and Groq's own per-minute token budget makes
every added call a real cost against a shared, already-strained quota tonight. Measuring
the two-hour cache expiry's effect on a cold first visitor, in scope per the task's own
text, not done: demonstrating it honestly means two real calls two hours apart, which this
session had no room for; the script is written so a later run can add it without changing
the method.

**Consequences:** nobody building on this project should reach for prompt caching as a
latency lever at this prompt size; the technique needs a materially longer cacheable
prefix to matter here; `Conversation.max_turns`, capped at eight exchanges, is the thing
that would have to grow before this is worth re-measuring.

## 2026-08-17: A disconnect mid-reply used to commit the whole reply as heard

**Context:** found writing M3.6's own test. `_play`'s `finally` decides whether to commit
a reply to history by checking `self._state.state is State.SPEAKING`, on the reasoning
that an interruption has already moved the state away by the time `_play` gets there.
That is true for `_interrupt`, which changes the state before it cancels playback, on
purpose, recorded in its own docstring. `run`'s own cleanup on a disconnect does not: it
calls `_stop_speaking` directly, and at that moment the state is still SPEAKING because
nothing else has touched it, which reads at `_play`'s finally exactly like reaching the
end on purpose. A turn cut off after one chunk of four was committed to history as the
whole reply, delivered.

The deeper reason this survived past the design that was meant to prevent it:
`SpeakingTurn.chunks()` delivers the same `None` sentinel whether `_pump` reached the end
of `answer` or was cancelled, deliberately, so a consumer's `async for` sees an ordinary
end of iteration either way. That is correct for `chunks()` itself, which only promises
audio in order. It means `_play`'s own loop finishing cannot be used as evidence of which
one happened, and nothing did.

**Decision:** `SpeakingTurn` gained a `cancelled` property, `self._task is not None and
self._task.cancelled()`, the one signal that actually distinguishes the two: asyncio
already tracks it on the task, `_pump`'s `except CancelledError: raise` is what leaves it
set, and no new state needs inventing. `_play`'s finally now requires `not
speaking.cancelled` in addition to the existing state check, so a disconnect stops
committing.

**Alternatives considered:** having `_pump` raise instead of delivering `None` on the
cancelled path, so `chunks()`'s own iterator would tell `_play` directly, rejected because
`chunks()`'s current contract, "ending when production finishes or is cancelled" per its
own docstring, is relied on by `_interrupt`, and changing it to raise on cancellation would
turn every existing interruption into an exception `_play` has to specifically not report
as a playback failure, trading one silent case for a noisier one elsewhere. Checking
`self._state.state` alone and accepting the disconnect gap, rejected once a test could
demonstrate it: M3.6's own acceptance is that an abandoned turn is recorded as abandoned,
and this was recording it as delivered.

**Consequences:** `tests/fault/test_disconnect.py` is the file M3.6 asked for, five tests,
including one that pins the opposite failure mode: a turn nothing interrupts must still
commit, so the fix for a false commit cannot become a false non-commit. `bench/waterfall.py`
had a related, narrower bug from the same investigation: `one_turn`'s own `wait_for`
raising `TimeoutError` skipped its `task.cancel()` entirely, leaving a session task
orphaned for however long its own retry loop took to give up, live once for 45 minutes.
Wrapped in `try/finally` there too, and a turn that times out is now recorded rather than
crashing the other nineteen.

## 2026-08-17: llama-3.3-70b-versatile is gone from Groq; moved to openai/gpt-oss-120b at low reasoning effort

**Context:** found live while resuming M4.3 measurement: every chat completion against
`llama-3.3-70b-versatile` returned HTTP 404, `model_not_found`. `GET /openai/v1/models`
confirmed it, listing no Llama chat model at all, only the two prompt-guard classifiers.
This is the model every turn of the deployed service uses; the live app was broken by a
provider-side catalog change, not by anything in this repo.

**Decision:** `openai/gpt-oss-120b`, the closest capability match still on the account and
verified live: correct streaming, correct structured tool calls with real arguments parsed
from a Hindi eligibility question. `llm/providers/quotas.yaml` now carries a
`default_params` field per provider, merged into every request body before any per-call
kwarg, because this model needed one: it is a reasoning model, and at the default effort a
60-token budget was spent entirely on the hidden reasoning channel and produced zero
content, confirmed live (`finish_reason: length`, `reasoning_tokens: 58` of
`completion_tokens: 60`). `reasoning_effort: "low"` fixed it, confirmed live with both a
plain reply and a tool call, and low effort is also the right default on its own merits:
every reasoning token spent before the first content token is time this project exists to
remove, and eligibility tool-calling does not need deep chain-of-thought to pick a scheme
id.

A second fix rides along, general rather than specific to this model swap. Any reasoning
model can spend its whole budget on the hidden channel with nothing left for an answer,
and the existing stream parser had no guard for it: `finish_reason: length` with zero
content and zero tool calls looks exactly like an ordinary, clean completion that had
nothing to say, `StreamCompleted` fires, and `_rounds` returns having yielded nothing, the
same silent empty reply this project has already found and fixed once from a different
cause tonight. `llm/providers/base.py`'s `_events` now raises `ProviderError` for that
specific shape, leaving every other truncation (real content already delivered, or a tool
call already assembled) untouched, tested in both directions.

**Alternatives considered:** `openai/gpt-oss-20b`, smaller and likely faster, not chosen
without a real comparison; a candidate for M4's own measurement rather than a guess made
under the pressure of a broken live service. `qwen/qwen3.6-27b`, also present on the
account and also plausibly a reasoning model with the same hazard, not verified tonight
and not chosen for the same reason. Raising `max_tokens` instead of setting
`reasoning_effort`, rejected because it does not bound the failure, only makes it rarer:
a harder question could still spend an arbitrarily large budget reasoning and never answer,
where a low reasoning effort setting keeps the channel short by design. Not adding the
`ProviderError` guard and calling the model swap alone sufficient, rejected on the same
reasoning `budget.py`'s own decision this session gives: the fix that only papers over the
one failure observed leaves the general shape of it live for the next provider change to
find again.

**Consequences:** `cerebras`'s and `openrouter`'s configured models were not re-verified
tonight; the same catalog-drift risk applies to them and neither is on the path this fix
needed to unblock. `default_params` is now a real, tested seam in the provider config, so
the next provider-specific request quirk has somewhere to live that is not a hardcoded
kwarg in `vaani/llm_turn.py`. Every live number in tonight's M4.3 run and everything after
it in M4 and M5 is dated against this model, not the one SPEC and BACKLOG still name in
older entries; a reader comparing this project's numbers to a Llama-3.3 benchmark is
comparing against a model this service no longer runs.

## 2026-08-17: An abandoned generator closes in whichever task garbage collects it, not the one that opened it

**Context:** found while running `bench/waterfall.py` across the full twenty-utterance
corpus for the first time. A single hand-picked entry measured cleanly; the full run
reported empty transcripts and zero reply text for every turn, with `llm.generate` and
`tts.synthesize` missing from the trace entirely by the last few turns. The proximate
symptom was `ValueError: ... was created in a different Context`, logged from inside
OpenTelemetry's own `context.detach`, and swallowed there rather than raised, so nothing
upstream ever saw it fail.

Every stage span in this pipeline (`STT_STREAM`, `LLM_GENERATE`, `TTS_SYNTHESIZE`) is a
`with stage_span(...)` block held open across the generator's own `yield`. That is
deliberate and correct for a generator that runs to exhaustion. It is not correct for one
that gets abandoned: `StreamingPipeline._listen` stops reading `ChunkedStt.stream` the
instant the final partial arrives, `bench/waterfall.py` cancels the whole session task
once a turn's audio has been sent, and `vaani.budget.speak_within` can be closed early by
a barge-in. An async generator left suspended mid-span does not close when its last
reader stops calling `__anext__`; it closes whenever Python's garbage collector or
asyncio's own asyncgen finalizer gets around to it, and that runs in a fresh
`contextvars.Context`, not the one the span was opened in. OpenTelemetry's context token
is only valid to detach in the context that attached it, so that later close fails, is
logged and discarded, and the span it belonged to never reaches the exporter. Enough of
these in a row corrupts the tracer's own context stack badly enough that unrelated spans
downstream stop recording too, which is what made the full run look so much worse than
the single-entry smoke test: one leak is noise, twenty compound.

A second, independent bug was layered on top while fixing the first. `speak_within` raced
only the first chunk of the answer against the filler deadline, via
`asyncio.ensure_future(anext(answer, None))`, then drove every chunk after that through a
plain `async for` on the caller's own task. That is the same shape of bug one level up:
the answer generator's span opens on whichever task ran its first `__anext__` and closes
on whichever task happened to be driving it when the generator finally exhausted, and
those are not reliably the same task. The obvious fix, one dedicated task driving `answer`
for its whole life, introduced a third bug: the task was awaited unconditionally in
`speak_within`'s `finally`, so closing `speak_within` early (an interruption) no longer
stopped `answer`, it just blocked the caller until the abandoned answer finished talking
to itself.

**Decision:** every generator in the pipeline that delegates to another async generator
via `async for x in inner(): yield x` now does so inside `contextlib.aclosing(inner())`.
Fixed at each hop: `StreamingPipeline._listen` and `RecoveringStt.stream` (both flagged in
the summary written before this session resumed), and, found while tracing the remaining
leak with the actual attaching and detaching task recorded on the token, four more layers
`_listen`'s own consumer chain runs through: `StreamingPipeline.run`'s consumption of
`speak_as_they_arrive`, `speak_as_they_arrive`'s consumption of both `from_stream` and
`tts.synthesize`, `from_stream`'s consumption of its `tokens` parameter, `FailingOverTts.
synthesize`'s consumption of whichever provider is live, and `StreamedTurn.run`'s
consumption of `_rounds`. `vaani.budget.speak_within` was rewritten around a dedicated
`_drive` task that owns `answer`'s entire lifetime and forwards chunks through a queue,
so the span always opens and closes on the same task regardless of how the caller times
the first chunk against the filler deadline; `_drive`'s own `finally` explicitly closes
`answer` too, since a task cancelled between two chunks (idle at a `yield`, not inside an
`await`) is exactly the same abandonment shape one level further in. `speak_within`'s
`finally` cancels `_drive` rather than awaiting it unconditionally, so an early close
actually stops the answer instead of outliving the thing that abandoned it.

A third, unrelated bug surfaced once the first two stopped hiding it: two of the twenty
corpus utterances are Devanagari script, and `bench/waterfall.py` had no encoding of its
own. Windows' default stream encoding when stdout is redirected rather than a real
console is the ANSI code page, which cannot represent Devanagari, so a log line
containing it raised `UnicodeEncodeError` from inside the session's own error-recovery
logging, which is a worse failure than the one it was trying to report. `bench/
waterfall.py` now reconfigures `sys.stdout`/`sys.stderr` to UTF-8 at the top when they are
not already, so a run of this script means the same thing on Windows as anywhere else.

**Alternatives considered:** catching and swallowing the context-detach `ValueError`
specifically, rejected because it treats the exporter losing spans as an acceptable
outcome rather than the bug it is, and does nothing about the resources (an open httpx
stream, a live Groq connection) the abandoned generator was still holding. Making
`stage_span` itself defensive about which context it detaches in, rejected as papering
over a real generator-lifetime bug at the one place that cannot tell whether the
generator was abandoned on purpose; `aclosing` fixes it at the point that has that
information; `stage_span` stays a plain, correct context manager. Leaving `speak_within`
awaiting `_drive` unconditionally and calling it fixed once the span leak stopped,
rejected after actually testing a barge-in against it and finding a turn that never
recovered until the abandoned answer finished on its own, which is the exact class of bug
this project's own LEARNING file (Spanlight's) says a green test suite will not catch on
its own.

**Consequences:** the full corpus run through `bench/waterfall.py` now reports zero
context-detach errors and zero encoding failures across repeated full passes, verified
directly by tracing which task attached and which task detached each context token, not
inferred from the absence of a log line. The remaining reason a full run can still show
empty replies is Groq's own per-account rate limit, tripped by this session's own repeated
testing (a live `RateLimitError` carried a 344 second `retry_after`), which is a real
quota constraint stated here rather than mistaken for a code defect a second time. Every
delegation-style generator added to this pipeline from here needs the same `aclosing`
treatment the moment it can hold a resource across a `yield`; nothing in the type signature
of an async generator marks that it needs one, so this is a pattern to watch for by hand,
not something a test can enforce structurally.

## 2026-08-17: The fixed corpus is synthesised, stated as a limitation, not passed off as recorded

**Context:** M4.2 and SPEC A8 both ask for the same thing: twenty fixed utterances, used
unchanged across every measured configuration, so a difference in the numbers cannot be
a difference in what was said. The backlog's own wording says "recorded." Nobody with a
microphone was available to produce that, and waiting on one would have blocked every
downstream measurement item, M4.3 through M5, on a step with no code in it at all.

**Decision:** `scripts/build_corpus.py` synthesises twenty utterances through the same
`EdgeTts` class the pipeline already uses for its own answers, decodes the MP3 output to
16kHz mono PCM16 with `miniaudio`, and writes both the WAV files and a manifest (id,
text, category, duration, filename) that `bench/waterfall.py` and anything after it read
rather than re-deriving. Verified against live Groq Whisper on a sample, not assumed: the
transcript comes back transliterated into Devanagari rather than matching the Romanised
input byte for byte, which is expected and is not the property being checked; the
meaning read back matched every sample.

`miniaudio` is a new dependency, added deliberately narrowly: a wheel-distributed MP3
decoder with no system binary, chosen after finding an `ffmpeg.exe` on this machine that
could have done the same conversion and rejecting it, because that binary came from an
unrelated LAMMPS installation rather than anything this project declares, and a corpus
that only rebuilds on this one machine is not a fixed corpus, it is a fixed corpus here.

**Alternatives considered:** waiting for a real recording session, rejected as blocking
the entire measurement milestone on a step that has no code and no clear date. Using
`pydub`, which also needs `ffmpeg` present on the machine running it, rejected for the
same non-reproducibility reason the found `ffmpeg.exe` was. Sourcing utterances from a
public Hindi speech dataset, rejected because those are not shaped to this project's
scenario classes, PM-KISAN and Ayushman Bharat by name are not going to appear in a
general-purpose corpus, and reshaping one would cost more than synthesising twenty
sentences purpose-written for this.

**Consequences:** every number `bench/waterfall.py` publishes from this corpus is
latency, not recognition accuracy, and has to be reported as such. A synthesised corpus
cannot say anything honest about disfluency, background noise, accent variation, or
microphone distance, because it has none of those; a claim that this project's numbers
generalise to real callers is not supported by this corpus alone and any report using it
has to say so rather than let the number imply otherwise. Recorded human speech, twenty
utterances hand-collected, is still the better version of M4.2 and is not what this is;
it is what made the other nine measurement items able to start tonight instead of
waiting on it.

## 2026-08-17: An explicit START while the agent is speaking now orders itself like every other interrupt

**Context:** found in the same pass as the two entries below, while a test for the new
`turn` span needed a case where `vaani.turn.interrupted` is true on a turn that still has
a valid backdated start. `_interrupt` calls `_begin_listening()` before
`_stop_speaking()`, deliberately: `TurnState.begin()` reads whether the turn it is
displacing was THINKING or SPEAKING, and the cancelled turn's own `_play` teardown reads
the state `begin()` just changed to decide whether to commit its reply as fully heard.
Get the order backwards and the teardown sees SPEAKING, commits the interrupted reply as
complete, and `begin()` computes `interrupted_previous` against a state that has already
moved past SPEAKING by the time it runs. `_on_control`'s handling of an explicit `START`
had exactly the backwards order, `_stop_speaking` then `_begin_listening`, so a START
arriving mid-answer committed the cut-off reply as whole and reported the new turn as
uninterrupted.

**Decision:** `_on_control` now calls `_interrupt()` for `START` when the state is
THINKING or SPEAKING, matching the ordering `_interrupt` already gets right, and keeps
the plain `_begin_listening()` path for the ordinary case, state already IDLE or
LISTENING. Conditional rather than always routing through `_interrupt`, to avoid sending
a second `READY` on every session's very first message: `run` already sends one
unconditionally before reading anything, and `_interrupt` sends its own at the end.

**Alternatives considered:** always calling `_interrupt()` regardless of state, rejected
for the double-`READY` reason above on the one path every session takes.

**Consequences:** the real client sends `START` exactly once, at socket-open, when state
is always IDLE, so this specific ordering was not reachable by anything a browser does
today; it is a latent defect closed before it could become live, the same shape of
finding as the two entries below.

## 2026-08-17: A second, uninterrupted turn never got its own AUDIO_START, silently, since the turn span was added

**Context:** building `bench/waterfall.py` needed real spans to read, which meant first
fixing M1.6's actual gap: `vad.endpoint` and `turn` were declared and defined but never
emitted (separate entry, this file). Testing the fix against a session with two ordinary,
uninterrupted turns rather than one exposed something the span work was not looking for.

`_send_audio` decides whether to send `AUDIO_START` by comparing `self._announced` to
`chunk.generation`, on the assumption that a new turn always carries a new generation.
That is only true across an interrupt. `TurnState.generation` advances exclusively inside
`begin()`, called only from an explicit `START` control message or from `_interrupt`;
an ordinary second question, asked after the first answer finished on its own with
nothing cutting it off, triggers `_begin_answering` again through the same endpoint-fires
path the first turn used, with the generation `_state` already had. So the second turn's
first chunk compared its generation against a value `_announced` already held from the
first turn, found them equal, and never sent `AUDIO_START` at all. The client's own
handler for that message is what opens a fresh playback buffer; without it, whatever
audio the server sent for that second turn had nothing telling the browser to expect it.

This was not reachable by a single-turn smoke test, which is every test this project had
run against a live browser so far, and it was not reachable by the fault-injection suite
either, which exercises failure inside one turn, not the ordinary case of a second one
following a clean first. It surfaced only because the two new spans needed a two-turn
session to prove `vaani.turn.index` actually counted past one.

**Decision:** `_begin_answering` resets `self._announced` to `None` at its own start,
once per turn, unconditionally. `_begin_answering` runs exactly once per new turn
regardless of whether the generation moved, which `chunk.generation` alone does not, so
this is the one place that reliably knows a new turn has begun. The comparison in
`_send_audio` is unchanged: `None != <anything>` is always true, so the first chunk of
every turn now announces correctly whether or not its generation is new.

**Alternatives considered:** tracking a separate per-turn identifier instead of reusing
`generation`, rejected as solving a problem `_begin_answering` already has a clean answer
to. Comparing against `self._turn_index` instead of `chunk.generation`, rejected because
`_send_audio` only has the chunk's generation to work with, and threading the turn index
through `AudioChunk` as well is a wider change for the same fix.

**Consequences:** every reply after the first one in an otherwise ordinary conversation
was very likely reaching the client with no `AUDIO_START`, which is a strong candidate
for why some live sessions have gone quiet after the first exchange with no error logged
anywhere: nothing failed, the message that tells the browser to expect audio just never
went out for a generation it had already announced once. No test caught it before this,
which is itself the finding worth keeping: this project's own suite had never once driven
a session past its first turn while checking what the client would actually have seen.

## 2026-08-17: The turn and vad.endpoint spans are backdated to the frame that made them true, not the frame that noticed it

**Context:** `bench/waterfall.py` needs real Spanlight spans to read, and M1.6's own
account of itself did not match the code: only four of seven declared spans were ever
actually emitted (`stt.stream`, `stt.request`, `llm.generate`, `tts.synthesize`).
`vad.endpoint` and `turn`, including the headline span this whole project's Proof
Artifact is measured against, existed only in `spans.py`'s CONTRACT and in
`bench/stages.md`'s prose. A waterfall built around that would have had no turn column.

**Decision:** `Endpointer` gained `speech_began_ns`, set on the first frame each
listening period that clears the speech threshold, `time.time_ns` rather than
`time.monotonic_ns` because `stage_span`'s `start_time` parameter is read by the same
clock OpenTelemetry itself uses to close a span, and the two clocks drift apart from
each other over a long-running process. `vad.endpoint` is emitted the instant `accept`
returns true, as a context manager entered and exited in the same statement, since both
of its boundaries are already in the past by the time anything can act on the true
result. `turn` shares the same start and is emitted once playback actually begins,
carried between the two points in `_begin_answering` and `_send_audio` by a small
`PendingTurnSpan` value rather than three loose attributes on `self` that would have to
be kept in step by hand.

`turn` has one case with nothing to backdate from: a verified interruption's audio is
written directly into the frame queue in `_interrupt`, bypassing `accept` entirely, so
the endpointer never saw it and `speech_began_ns` stays `None`. That turn's span is
skipped, logged, and not fabricated from a nearby timestamp. `playback.first_audio`
stays entirely unemitted, for the reason M1.6's own entry already gave and never
resolved: it is supposed to close on a browser acknowledgement the client does not send,
the same gap blocking M2.15's spoken half, and a span invented to have a plausible
zero-ish duration would be worse than one honestly missing.

**Alternatives considered:** deriving `turn` and `vad.endpoint`'s timing by re-reading
`TurnClock`, which already backdates to the last frame of speech, rejected because
`turn`'s own definition in `bench/stages.md` starts at the *first* frame, not the last,
and reusing a clock built for a different boundary would have quietly measured the wrong
thing while looking correct. Fabricating a `turn` start for the verified-interruption
case from `BargeClock`, which does have a valid backdated start, rejected as a
monotonic-to-wall-clock conversion whose correctness depends on the process never
adjusting its own clock mid-session, a caveat real enough that `Interactivity` already
carries a version of it; skipping the span is the version of this decision that does not
need to trust that.

**Consequences:** five new tests in `tests/vaani/test_session.py` prove both spans
against a running session rather than only against the contract: one ordinary turn, a
second turn's index incrementing, a turn restarted by an explicit `START` correctly
marking `vaani.turn.interrupted`, and a verified interruption's turn span confirmed
absent rather than merely unchecked. Building the second-turn test is what found the
`_announced` bug recorded separately above; the span work and that fix are effectively
one session's worth of the same investigation.

## 2026-08-17: A quiet caller gets told the mic looks fine, in text, and the spoken version is deliberately not built tonight

**Context:** M2.15 asks for Bolna's check-in message after extended silence, distinct from
the S6 microphone diagnostic, on the grounds that "the mic looks broken" and "the mic is
fine but nobody is talking" currently read identically to the person on the other end.
`Endpointer.diagnose()` already separates `SILENT` (digital zero, likely muted or denied
permission) from `TOO_QUIET` (something registers, never enough to count as speech), so the
signal already exists; only `TOO_QUIET` gets an additional message, since `SILENT` already
has a specific, accurate diagnostic and "are you there" is the wrong question for someone
who cannot be heard at all.

Speaking it rather than showing it, which is what the item and Bolna's own design both
mean by "say," was scoped out once the client's actual message handling was read rather
than assumed. `case "audio_end"` calls `newTurnSoon`, and if `sawAnswer` was never set true
by a `transcript` message it shows "that was filler, the answer did not arrive." A check-in
played through the existing `audio_start`/bytes/`audio_end` sequence sends no transcript, so
it would trip that fallback and misreport its own message as a failed turn. Fixing that
correctly needs the protocol to say what kind of audio is arriving, which is a real but
separate piece of work, against a client with zero existing test coverage for inbound audio
playback, in the same session that had already found two other live-only bugs in exactly
that path.

**Decision:** `ServerMessage.CHECKING_IN`, a text-only control message, sent once per
listening window when `diagnose()` reads `TOO_QUIET`. It reuses S6's existing 5-second
`silence_timeout_ms` signal rather than Bolna's separate 6-second timer, since both are
answering the same underlying question, "has this gone on too long with nothing," and a
second independent clock for a nearly identical threshold is complexity with no behaviour
to show for it. The client gets one new, purely additive `case` in its message switch,
which cannot alter any existing message's handling because it is not touching one.

Writing the test for this surfaced a real, narrow, pre-existing gap: `diagnose()` reads
`peak_rms`, a running maximum updated on every frame, while `started` needs
`min_speech_ms` of sustained speech to flip. Real speech beginning right after a long
silence is loud on its first frame and not yet `started` for the next several, so for that
window the state genuinely reads `TOO_QUIET` while somebody has already begun talking. Left
in `diagnose()` itself, this would have greeted the start of a real answer with a "still
there" message. Guarded at the call site instead, `speech_ms == 0` alongside the state
check, because fixing `diagnose()`'s own timing is a separate, more invasive change than
this item asked for, and the guard is enough to make this feature correct without it.

**Alternatives considered:** a synthesised, spoken check-in via the existing filler
mechanism, rejected for the `sawAnswer` reason above; the fix belongs in the protocol, not
smuggled through a mechanism built for something else. A second independent silence timer
matching Bolna's exact 6 seconds, rejected as a distinction with no real difference from
reusing S6's 5-second one, for a session that has no evidence either number is closer to
right, since M4.2's corpus has not run.

**Consequences:** the spoken half stays open, tracked as the remainder of M2.15, and needs
its own careful pass: a protocol field distinguishing check-in audio from answer audio, a
client-side test harness for inbound playback that does not currently exist at all, and
verification against the live deploy before it ships, not blind trust that it works. The
`diagnose()` timing gap is now known and stated rather than quietly worked around; whatever
else reads `TOO_QUIET` in the future inherits the same narrow window unless it is fixed at
the source.

## 2026-08-17: A resolved interruption is acknowledged, because the hazard that blocked it cannot reach that path

**Context:** M2.8 had its recording half done since M4.12 and its acknowledgement half
deliberately left unbuilt, for a stated reason: speaking "haan, boliye" while the user is
still mid-sentence is the agent talking over them, `agent_interrupted_user`, and the item's
own note said this is only safe once the utterance has ended. Revisiting it meant
confirming that boundary in the code rather than trusting the note. `_commit_barge`'s
`complete=True` branch, the one that starts a new turn immediately, is reachable only from
`_resolve_barge`, which itself only runs once the interrupting utterance's own silence has
been detected: the duration path commits with `complete=False` and keeps listening instead.
So a turn that begins by acknowledging has, by construction, never been triggered while
somebody was still talking. The hazard is not mitigated here, it does not reach this code
path at all.

The infrastructure this needed was already sitting unused: `Purpose.RESUMING` and its four
committed clips exist from M1.13, referenced nowhere.

**Decision:** `VoiceSession`'s `filler` constructor argument changes from a fixed
`Callable[[], AsyncIterator[bytes]]` to `Callable[[Purpose], AsyncIterator[bytes]]`.
`_begin_answering` takes a new `acknowledge` keyword, set by `_commit_barge` on the one
call site that can reach it, and `_audio` picks `Purpose.RESUMING` over `Purpose.THINKING`
from it. `app/routers/voice.py`'s `speak_filler` takes the same purpose and its on-demand
synthesis fallback now keys `FILLER_TEXT` by purpose too, so a fork without the bank still
says the right phrase rather than always the thinking one.

**Alternatives considered:** a second, separate acknowledgement callable on `VoiceSession`
alongside the existing filler one, rejected as two things to keep synchronised for what is
the same mechanism asking a different question of the same bank. Deciding the purpose
inside `speak_filler` itself by inspecting session state, rejected because the filler
function has no session to inspect by design, an intentional seam that lets it be tested
and swapped without a `VoiceSession` in scope at all; the caller already knows which turn
it is starting and is where the decision belongs.

**Consequences:** every test that constructs a `VoiceSession` with a hand-written filler
fixture needed its signature widened to accept the purpose argument, five call sites across
`tests/vaani/test_session.py`, `tests/fault/test_silent_mic.py`, `tests/test_redaction.py`,
`tests/app/test_voice_socket.py`, and `bench/bargein.py`, none of which care what the
argument is and all of which now accept and ignore it. Two new tests exercise the actual
property: one drives a verified interruption through a filler that records every purpose it
was asked for and asserts the second turn's is `RESUMING`, the other drives the duration
path the same way and asserts `RESUMING` never appears in the record at all.

## 2026-08-17: Every provider gets one connection for the process's life, not one per request

**Context:** picking up M0.6, the literal task was narrower than what reading the actual
request path found. `OpenAICompatibleProvider` and `GroqWhisper` both fall back to a fresh
`httpx.AsyncClient()`, opened and closed, whenever no client is injected, and nothing in
`llm/providers/registry.py` or `app/routers/voice.py` ever injected one for the real
(non-test) providers. Every provider call therefore paid its own TCP and TLS handshake to
a host that is not local, on every request, for the whole session, not only the first
turn's. `ChunkedStt` sends five to eleven of these per utterance by design (SPEC A4), so
this was not a one-time cost at socket-open, it was a per-partial cost repeated all
session long.

Measured before deciding it mattered: a four-second utterance's seven STT requests, same
audio, same 600ms interval, took 4.20s wall time with a fresh client each call and 2.57 to
2.88s reusing one. Roughly a third of the STT stage's wall time, on top of what the
interval widening the night before already cut, for a change that touches two constructor
call sites and adds no new request anywhere.

**Decision:** `load_providers` in `llm/providers/registry.py` now constructs one
`httpx.AsyncClient` per configured provider and passes it in, so every `OpenAICompatibleProvider`
holds a persistent connection for the process's entire life; `_PROVIDERS` is already a
module-level singleton built once at import, so this costs nothing extra to wire in. A
matching `_stt_client` singleton in `app/routers/voice.py` is passed into the one
`GroqWhisper` a session constructs, covering the chunked partials, the batch fallback, and
the backchannel check, since all three already share that one instance. Safe across
sessions despite being one shared object, because `_in_session` already guarantees only one
session runs at a time, so the connection pool is never contended.

**Alternatives considered:** a throwaway warmup request fired at socket-open, the literal
reading of M0.6's own text. Rejected because it spends real provider quota, Groq's or
whichever's, on every accepted connection including ones that never send a frame, which on
a public endpoint with no rate limiting yet (M3.8 is still open) is quota an uninterested
visitor or a bot can spend. A persistent client costs nothing until a real request needs
it and never fires speculatively. Session-scoped clients, opened at socket-accept and
closed at socket-close rather than process-lifetime singletons, rejected as strictly worse
for no safety gained: `_in_session` already serialises every session, so a process-wide
client is exactly as safe and additionally warm for the very first session after a cold
start, which a session-scoped one is not.

**Consequences:** two tests close the actual gap rather than only asserting the change
compiles: `test_load_providers_gives_each_real_provider_a_persistent_connection` reads the
provider's own `_client` attribute after construction, and
`test_an_injected_client_survives_the_call_for_reuse` calls `GroqWhisper.transcribe` twice
through one client and asserts it is still open and still usable after both, which is the
property the whole fix depends on and nothing before this proved. Six providers
(`gemini`, `groq`, `cerebras`, `openrouter`, `ollama`, `sarvam`) each now hold an idle
client object at all times regardless of whether this process ever calls them; harmless,
since construction alone does not dial out and the memory cost of an idle client is
negligible against the 512MB instance this runs on.

## 2026-08-17: An unanswered user turn is merged with what follows, not left as a second question

**Context:** live testing kept producing replies that answered only half of what was
asked. Traced to `Conversation`'s own invariant rather than guessed at: `messages`
alternates user and assistant roles by construction once a turn is actually answered,
because every path that commits or truncates a reply (`commit`, `truncate` in
`vaani/history.py`) appends an assistant message right after the user's. So two user
messages appearing in a row can only mean the first one never got any reply recorded at
all, which happens when an interrupt lands before the model ever staged a sentence, or
after it staged one but before any of its audio went out (`truncate` drops that case
too, per the guard already in place). A user cut off mid-thought who then keeps talking
produces exactly this: the first half sits in history unanswered, the second half arrives
as what looks like a fresh, unrelated question, and the model reads them as two separate
things rather than one continued one.

This is Bolna's `pop_and_merge_user`, and the backlog had already named it as owed
(M1.17) before tonight's testing made it visible.

**Decision:** `user_said` now checks whether the previous message is a bare, unanswered
user turn, and if so replaces it with the two texts joined rather than appending a second
one. The existing overlap check (a reused STT partial and the final that follows it
sharing a prefix) runs first and is unchanged; this is the case where the second
transcript does not extend the first, it continues it.

**Alternatives considered:** popping the old message and inserting a merged one, closer
to Bolna's literal mechanism, rejected as equivalent in this record's append-only shape
and more code for the same result: `messages[-1] = ...` is the pop and the insert in one
step. Requiring the caller to say explicitly whether a transcript is a continuation,
rejected because the caller (`vaani/session.py`) has no better signal than the record
already has: the alternation invariant is the signal, and asking for a second one invites
the two disagreeing.

**Consequences:** an existing test's fixture predated this and could not tell a
genuinely new follow-up question apart from an interrupted one, because it called
`user_said` twice with nothing between them, which is indistinguishable from the bug at
the object's own invariant. Rewritten to commit a real reply between the two calls,
which is what makes the two cases look different. Two new tests cover the merge itself,
including the dropped-not-committed path, so both ways an assistant message can fail to
appear are exercised rather than only the more obvious one.

## 2026-08-17: A tool call written as prose is filtered out of the reply, not swallowed with it

**Context:** verifying the earlier tool-schema fixes against live Groq, one run in five
produced a reply with the model's own tool-call syntax embedded in it as ordinary text:
`"...eligibility check karne ke liye, <function=check_eligibility>{"applicant":
{"annual_income_inr":"50000"}}</function> ka use karna hoga."` Groq's structured
`tool_calls` mechanism has a known failure mode where Llama writes the call inline in
`delta.content` instead, and nothing in `_events` treats that differently from real
reply text: it would have been synthesised and spoken verbatim, unpronounceable syntax
and an applicant's own income both included.

The marker cannot be matched against one chunk at a time. Groq streams text a handful of
characters per delta, so `<function=` arrives split across chunk boundaries in
production, and a check that only looked at each chunk in isolation would miss it on
exactly the boundary a fixed test fixture would not, by luck, ever reproduce.

**Decision:** `ToolCallLeakFilter` in `vaani/llm_turn.py`, a small streaming state
machine fed one chunk at a time. It holds back only the shortest suffix that could still
become the marker's prefix, releases everything else immediately, and once the marker is
confirmed, buffers and discards up to the matching close tag rather than the rest of the
round. Real prose before and after the tag survives; only the tag and its payload are
removed. A leak that never closes, the round ending mid-tag, is dropped entirely on
`flush` rather than spoken as a truncated fragment.

**Alternatives considered:** degrading the whole turn to `COULD_NOT_CHECK`, the path
`M3.10`'s acceptance line originally described and the same one an uncaught
`ProviderClientError` now takes. Rejected once the live example was in hand: it had real,
useful reply text on both sides of the leaked tag, and discarding all of it over one
provider-side formatting slip trades a mostly-working answer for a canned apology.
Matching against the whole accumulated response text at the end of the round rather than
streaming, rejected because it delays every chunk in the round behind the last one,
which defeats first-sentence flush for every reply, leaked or not, to guard against a
leak that occurs in roughly one call in five.

**Consequences:** the filter is per round, discarded and rebuilt fresh each time,
because a leak is confined to the text one round produced and carrying state across
rounds would buy nothing. Tested against every possible chunk width from one character
to the whole string on the real leaked text, not a hand-picked split, plus a round trip
through `StreamedTurn.run` itself so the wiring is proven, not only the class in
isolation.

## 2026-08-17: A dead connection now releases the one-at-a-time lock instead of holding it until the process is restarted

**Context:** live testing hit "busy: One session at a time on the free tier" on every
attempt, on a freshly restarted process, within two seconds of startup, before any
legitimate traffic could plausibly have arrived. Traced through Render's logs: 22
WebSocket connections opened over three hours with zero recorded closes. `_in_session`
in `app/routers/voice.py` is a bare `asyncio.Lock` held for the duration of
`session.run()`, and `VoiceSession.run()`'s receive loop awaits `Transport.receive()`
with no bound. A connection that never sends a clean close, a dropped link, a frozen
tab, a laptop put to sleep, leaves that await pending for as long as the underlying
socket takes to notice, which is governed by the OS's own TCP timeout rather than
anything this process controls. One such connection holds the lock indefinitely, and
every visitor after it is told the service is busy until somebody restarts the process
by hand, which is what tonight's testing needed twice.

**Decision:** `VoiceSession._receive_or_give_up` wraps the transport's `receive()` in
`asyncio.wait_for` at `IDLE_RECEIVE_TIMEOUT_S`, twenty seconds, and a timeout is treated
exactly like a disconnect: `run`'s existing `finally` clause already abandons a
half-finished turn and returns, which releases `_in_session` the same way a clean close
does. Twenty seconds is safe rather than tight: a genuinely connected, listening client
is never silent this long, because the browser's `AudioWorklet` forwards a frame every
20ms for as long as the tab is capturing, mid-answer included, since barge-in detection
depends on that same stream continuing to arrive during playback. Going quiet for a
thousand times that interval is not a pause, it is an absence.

**Alternatives considered:** putting the timeout in `SocketTransport.receive()` instead,
which is Starlette-specific and would need a new fake-socket test harness; the session
level reuses the existing `FakeTransport` fixture and applies to any transport
implementation, not only the real one. A lease with a heartbeat the client renews,
rejected as more protocol for the same outcome: the existing frame stream already is a
heartbeat, twenty milliseconds at a time, and nothing else needs to be invented to read
it as one.

**Consequences:** `M3.6`'s own acceptance criterion is about the client's reconnect
experience and is not fully met by this; what changed is that the server side of a dead
connection now heals itself rather than requiring intervention. `M3.8`'s "busy" refusal
already worked as designed and had no test exercising the liveness gap, because nothing
in that milestone's acceptance asked for one; this was found by running the live
service, not by a test that was owed and missing.

Wrapping every `receive()` in `asyncio.wait_for` shifts scheduling enough that it also
surfaced a real, pre-existing race in
`test_the_interrupting_audio_starts_the_turn_it_caused`: `_interrupt` sets
`interrupted_previous` before it awaits `_stop_speaking`, and the carried-over frames are
only queued after that await completes, so a check gated on the flag alone can sample the
queue in the gap between the two. Measured before concluding anything: 8 clean runs of
that one test with the wrapper removed, flakes reintroduced the moment it went back.
Fixed at the test, which now waits for the frames themselves rather than a precondition
that can be true slightly before they are, confirmed clean over 20 repeats and three full
suite runs. The lesson is not really about this one test: a change to scheduling can turn
a latent race already sitting in the suite into a failure that looks like it belongs to
the change, and the fix belongs wherever the actual race is, which examining the failure
rather than reverting on sight is what found.

## 2026-08-16: The chunked STT interval moves from 400ms to 600ms, measured against live Groq rather than reasoned about

**Context:** a live test reported "very very high latency." Pulled the actual session from
Render's logs rather than guessing: one turn took roughly 7.5 seconds end to end, and about
5 of those seconds were the STT stage alone, nine sequential Whisper requests
(`stt.stream_done requests=9`) between the first partial and the final transcript. `ChunkedStt`
re-sends the whole growing buffer every `interval_ms` because Groq's endpoint takes a finished
file rather than a stream, and the class's own docstring already named this "likely the single
largest difference between the two waterfalls." Nobody had put a number on it before this test.

Before changing the interval, the real risk was whether widening it could make an ordinary
turn's endpoint wait worse, since partials also feed `Endpointer.note_partial`, which shortens
the trailing-silence wait from 700ms to 200ms once a partial looks complete. Traced rather
than assumed: `silence_ms` counts continuously from the last frame of speech regardless of
when a partial arrives, so a partial confirming completeness at `silence_ms=X` satisfies
`silence_ms >= early_silence_ms` immediately if X already exceeds it, firing on the very next
frame. The endpoint fires at `max(X, early_silence_ms)`, which is at most `trailing_silence_ms`
either way. A later-confirming partial is therefore never worse than the unconditional
700ms fallback the turn would have had without semantic endpointing at all, only sometimes
less early. That made the interval a one-sided knob to widen, not a trade against the
endpoint-wait number, which is the reasoning `bench/waterfall.py` will eventually have to
make a stack-wide claim rather than a stage-local one.

**Decision:** `interval_ms` defaults to 600. Measured against live Groq for a four-second
utterance: 11 requests and 7.76s of STT wall time at 400ms, 7 requests and 4.05s at 600ms, 5
requests and 3.25s at 800ms. 600 over 800 because the marginal win past it is small (4.05s to
3.25s) while the cost is not: a wider interval delays how soon a completed sentence can be
confirmed, bounded by `trailing_silence_ms` but real, and 800 would double that delay against
400 for a small further cut in requests.

**Alternatives considered:** 800ms for the larger raw request-count cut, rejected for the
diminishing-returns reason above. A provider switch to a genuinely streaming STT endpoint
(Sarvam's Saaras is in the provider table already), rejected for now because it spends credit
that QUOTAS.md records as scarce and the dev loop is built to never touch it; also because 600
already moves the dominant cost without spending anything, and switching providers is a
question this project's own two-stack ablation should answer with a measurement rather than a
reaction to one anecdote. Not changing anything and waiting for M4's waterfall, rejected
because the cost was already visible, named in the code before this session, and measurable
in five minutes against the real endpoint.

**Consequences:** every eligibility question now pays roughly half the STT wall time a long
utterance did before, and short utterances pay at most 200ms more before a completeness
check can fire, bounded by the reasoning above. `tests/vaani/test_stt_stream.py` pins the
default so a future edit changes it on purpose. What is still not measured: whether 600 is
actually the point of diminishing returns for the full range of utterance lengths real users
produce, rather than the one four-second sample here, which is exactly the gap `bench/waterfall.py`
and the recorded corpus in M4.2 exist to close, and this decision does not pretend otherwise.

## 2026-08-16: A schema a provider validates has to accept what the model actually sends, and a rejection it cannot see must not be silent

**Context:** every eligibility question on the deployed service failed: the user heard
the filler clip and nothing after it. Reproduced against the live socket and traced
through Render's logs. The model, llama-3.3-70b on Groq, called `check_eligibility`
with `annual_income_inr: "50000"`, a string, against a schema declaring `integer`.
Groq validates tool arguments against the declared JSON schema on its own side before
generation completes, rejected the call, and streamed back HTTP 200 with `event: error`
and no `choices` at all. `_events` only inspected `choices`, saw an empty stream, and
raised `"stream ended before any content"`, the message written for a dropped
connection. A schema rejection and a dead socket were indistinguishable, and the
provider's own explanation, which named the exact field and the exact type mismatch,
was discarded before anything could read it.

Fixing the type surfaced a second failure on the very next call: `find_schemes` was
rejected the same way over `limit`. The model sends every number as text, not just the
two in the first report. And once both were fixed, a live question exposed a third
failure that could not have been reached by a fix to our own schema: the model asked
about "PM Kisan" by name, called the tool with `scheme_id: "PM Kisan"` rather than the
slug `"pm-kisan"`, and `ToolError`'s message named the complaint but not the fix, so
three rounds of guessing produced three more wrong ids and the turn gave up. And a
fourth, found only by re-running the live question repeatedly: Groq can fail to parse
its own generation into a structured tool call at all (`tool_use_failed`, "Failed to
call a function"), which raises before `ToolCallsRequested` ever fires and has no
`tool_call_id` to attach a corrective result to. That exception was uncaught anywhere
in `StreamedTurn`, so it propagated past every degradation rule in SPEC to the
session's generic playback handler, which is the same filler-then-silence experience
from an entirely different cause.

None of the four were visible to the test suite. The tool tests hand
`check_eligibility` a dictionary directly, which never builds the schema, never sends
it anywhere, and never meets the validator that actually rejects it - the arguments
under test were the ones this repo would have written, not the ones a model writes.

**Decision:** four independent fixes, one per failure.

`Rupees`, `Acres`, and `Limit` in `vaani/tools.py` are `Annotated` types carrying a
`BeforeValidator` that coerces a numeric string (commas and a rupee sign stripped,
since a model asked for an income in a Hindi conversation writes one) and a
`WithJsonSchema` override publishing `{"type": ["integer", "string"]}` rather than a
bare `"integer"`. The runtime type is unchanged: coercion happens before Pydantic's own
`int`/`float` validation, so `check_eligibility` still cannot be handed a string, only
the door into it widened. Plain `Annotated` aliases rather than a PEP 695 `type`
statement, because the alias form emits a `$ref` into `$defs` and this schema is read
by a provider's validator rather than by us: a reference it does not resolve is a worse
contract than a repeated few lines. `OptionalText` does the same for `state: str | None`,
which the model fills with the literal string `"null"` when it has nothing to put
there, silently filtering a search by a state of that name rather than leaving it
unset.

`llm/providers/base.py` parses an `error` key on any decoded SSE frame and raises
`ProviderClientError` with the provider's `code` and `message`, never
`failed_generation`, which holds the arguments the model tried to send and on this
pipeline is an applicant's income.

`check_eligibility`'s `ToolError` on an unknown id now lists every valid `scheme_id`,
because that message is the only channel the model has to correct itself inside
`MAX_TOOL_ROUNDS`, and a scheme's name is not its slug.

`StreamedTurn._rounds` wraps each round's stream in a `try`/`except ProviderClientError`
and degrades to `COULD_NOT_CHECK`, the same message and the same tested path that
running out of tool rounds already uses. Chosen over retrying blind, since there is no
corrected information to feed back and a graceful, known failure path already exists
for the same words a listener would hear either way.

**Alternatives considered:** an enum of valid scheme ids enforced in the published JSON
schema, so Groq itself rejects a wrong id, rejected because that rejection would arrive
as a `ProviderClientError` with no `tool_call_id`, exactly the fourth failure this
decision already had to add a path for, and turning a self-correctable mistake into an
uncatchable one is the wrong direction. Retrying a `tool_use_failed` blind within the
round budget, rejected as the same wait for a worse expected outcome the whole verified
barge-in arm exists to avoid paying elsewhere: the failure is a parsing quirk in the
provider's own generation, and nothing this process does changes the odds on a retry
sampled from the same distribution.

A fifth failure was found and left open: in one of five live runs, the model wrote its
tool call as literal text in `delta.content` - `<function=check_eligibility>{...}
</function>` - rather than a structured delta, which every check here passes and which
would be synthesised and spoken verbatim, including the applicant's own figures inside
garbled syntax. Not fixed here, because catching it needs buffering partial content
across chunk boundaries to match a pattern reliably, and a pattern wrong in the other
direction would swallow a legitimate reply, which is worse than the flake it replaces.
Recorded as M3.10.

**Consequences:** `tests/vaani/test_tool_schema_contract.py` enumerates every numeric
field in every published tool schema rather than checking the two that were reported,
because fixing exactly the two moved the failure to a third the same afternoon.
`tests/llm/test_streaming.py` and `tests/vaani/test_llm_turn.py` each gained a test
built from the exact frame Groq sent, not an imagined one. None of the four call Groq:
what they prove is that the code handles the frame correctly once it exists, not that
the frame does. That gap is real and is what running the live probe caught that the
suite could not, three separate times in one evening, which is the same lesson
Spanlight's field study paid for: an eval set, or here a test suite, whose inputs
cannot move passes forever.

## 2026-08-14: Barge-in latency is measured against a modelled browser, and the clock is checked against the frame it infers

**Context:** M2.4 asks for barge-in latency over 20 runs, and `bench/stages.md`
had already split it into three numbers rather than one, because speech over the
agent is a hypothesis before it is a decision. The definition of the first of
them is where the difficulty is: audio stops reaching the listener at the earlier
of the browser pausing on its own level detector and the server's PAUSE arriving.
The browser half happens in the browser, is never reported, and is therefore
invisible to any clock inside the server. A client that did report it would be
reporting across two unsynchronised clocks, so the reported number would be the
skew between two machines plus the thing being measured.

**Decision:** `bench/bargein.py` runs the page's own preemption rule, three
consecutive frames over an RMS of 500, on the same frames it feeds a real
`VoiceSession`, in one process against one monotonic clock. Both ends of the
number are then observable and the earlier of them is `paused_ms`. Three arms are
driven separately and reported separately, because the duration path waits out
800ms of speech and the verified path waits out an utterance and a transcription,
and a median over both would be a median over two distributions.

The start boundary is inferred rather than observed by the running system, and
that inference is what `BargeClock` exists for. Detection cannot fire until
`min_speech_ms` of speech has arrived, so the clock is wound back by the length of
the unbroken speech run to reach the frame the speech began on. The run rather
than the total: a chair creak two seconds earlier adds one frame to `speech_ms`
and no time to the run, and backdating by the total would report the whole
intervening silence as barge-in latency.

Nothing in production can check that inference, because production has no record
of which frame the speech began on. The harness sent it, so the bench publishes
the disagreement between the two as a `clock error` row.

**Measured, 20 runs per arm, on the development machine, frames paced at 20.7ms
against a 20ms target:**

| arm | paused p50/p95 | server PAUSE p50/p95 | settled p50/p95 |
|---|---|---|---|
| duration | 61.2 / 62.6 | 207.1 / 209.4 | 831.9 / 836.3 committed |
| verified, real interruption | 61.6 / 62.4 | 206.8 / 208.8 | 1141.9 / 1147.7 committed |
| verified, backchannel | 61.5 / 62.5 | 207.3 / 209.9 | 1139.5 / 1149.8 resumed |

The client's detector won every single run of all sixty, at roughly 61ms against
the server's 207ms, and that gap is before a network is involved at all. On the
deployed stack the server's PAUSE is a further one-way trip behind. That is what
M2.11 buys, and it is now a number rather than an argument.

Almost all of each figure is the threshold it waits for. Above the frame of audio
that first made each decision possible, the client pause costs 0.0ms, the server
pause 0.4ms at p50, and abandoning the turn 1.6ms. So there is nothing to optimise
here and the knobs are the whole latency: `PREEMPT_FRAMES`, `min_speech_ms` and
`commit_ms` are what the ablation should sweep, not the code between them.

None of these are compared against the 100ms figure in the tail-control brief.
That remains a budget to verify against now that a measurement exists, in that
order, because this portfolio has twice paid for the other one.

**Alternatives considered:** having the client report its own pause over the
socket, rejected for the clock-skew reason above and because it puts a message on
the wire that exists only to be measured. Measuring the server's PAUSE alone and
publishing it as the barge-in latency, rejected as the flattering-in-reverse
option: it triples the number a listener actually perceives, and it would have
made client-side preemption look worthless. Driving a real browser, rejected
because a test that needs a microphone is a test that never runs in CI, and this
one has to run there.

**Consequences:** the bench duplicates a rule that lives in `web/index.html`, so
a change to the page silently changes what this measures. There is a test that
reads both constants out of the page and fails when they drift, which is the
price of the duplication paid rather than deferred.

Two facts about the harness are now part of the published numbers. Frames are
paced against an absolute schedule and never sent sooner than one interval after
the last, since a catch-up burst is speech arriving faster than anyone can talk
and it put the client's own pause 56ms after an onset its three-frame detector
cannot beat 60ms on. And the Windows default timer period of 15.6ms is most of a
frame, so the bench raises it to 1ms for the duration of a run: 22.3ms mean and
35.7ms worst before, 20.7ms mean and 20.9ms worst after. The residual 3.5% is
reported at the top of every run rather than corrected away, because it is in
every number underneath it.

The clock error is published raw and net of that pacing. Raw it is 6.5 to 10ms;
net it is 0.1 to 0.2ms, which is what says the backdating is right. Only the net
figure is asserted on, because a test that fails whenever the machine running it
is busy is a test that gets disabled instead of fixed.

Nothing in the regression test is asserted in milliseconds of wall clock either,
and that was not the first version. Every threshold in this pipeline is counted in
frames of audio, so the earliest moment a decision can be reached is the moment its
nth frame arrived, and a bound written against a nominal 20ms fails whenever the
machine cannot feed audio that fast. Measured at 20.7ms a frame idle and 28.0ms
under a full suite, which moves the headline pause from 61ms to 84ms with no code
change at all. The bench now says at the top of every run whether it stayed inside
a 10% tolerance, and reports the overhead columns, which survive a slow run,
alongside the absolute ones, which do not.

## 2026-08-14: The conversation record admits a sentence only once its audio has gone out

**Context:** M2.7b, and the fact that Vaani had no conversation record at all. Each turn
built `[system, user]` fresh, so the agent could not refer to anything it had said. Adding
history is the obvious next step and it is also the moment the bug M2.7 exists to prevent
gets designed in or out, which is why the two were done together rather than in the order
they were written down.

**Decision:** an assistant turn is *staged* when the model produces it and *committed* only
when its audio has reached the transport. Text whose audio was blocked or abandoned never
enters the record, so the ordinary interruption is handled by not writing rather than by
repairing. Reconstruction is the fallback for the single turn that was half spoken: the
fraction of the reply's audio that had played, applied to its characters, trimmed back to
a word boundary.

That ordering is the half of Bolna's design two earlier readings missed, because it is not
in `sync_history` at all. It is `_stage_assistant_history` and its pair, forty lines away
and much simpler than the mechanism they exist to make rare.

**The split of responsibility is the part worth stating.** The pipeline reads the record to
build the prompt; the session performs every write. A write needs the generation, and the
transport is the only thing that knows whether a sentence's audio actually went out. This
needed no change to the `Answer` protocol, because the session's report callbacks already
close over the generation, which is a seam that turned out to be load-bearing for a reason
it was not built for.

**Alternatives considered:** committing when the model produces the text, which is what
every implementation does by default and is the bug. Carrying a sentence index on every
audio chunk so each sentence could be committed exactly when its own audio was sent, which
is more precise and costs a type change through `speak_as_they_arrive`, `SpeakingTurn` and
the `Answer` protocol; the proportional estimate is within a word of the same answer and
needs none of it. Word marks from the synthesiser and a played-position report from the
client, which the original M2.7 assumed were required and which the playout estimate makes
unnecessary.

**Consequences.** Four guards, three read off Bolna and one from our own shape, each of
which is a bug the record would otherwise ship: refuse to trim without evidence, or a
second cleanup deletes a fully heard reply; never trim an already-committed turn, because
cleanups run more than once and the later one knows less; count answer audio only, since
filler plays *before* the answer and counting it pushes the estimate past the end; and drop
rather than commit empty, because an empty assistant message is not the same as no message
and a model reads it as a refusal.

History goes after the system prompt and before the user turn, which is the ordering that
keeps the static prefix cacheable. Groq caches automatically on supported models, so
putting history first would give up about half the input cost and a slice of time to first
token without anything in the code looking wrong. The record is bounded at eight exchanges,
because the prompt grows with the conversation and this runs on a free tier.

`llm_turn` now takes history, which means the unstreamed baseline in `vaani/turn.py` and
the streamed path can be given the same conversation, so the ablation is not comparing a
stateless arm against a stateful one.

## 2026-08-13: The model may phrase an eligibility answer and may not introduce a number

**Context:** M3.9 and M1.10, which turn out to be one decision from two ends. `vaani/tools.py`
refuses to invent a threshold, and nothing stopped the model stating one in prose, which is
what gets synthesised and played into somebody's ear. At the other end, the applicant's own
figure arrives as speech: "pachaas hazaar" read as 50 rather than 50000 moves them across an
income limit, and a model asked to both normalise that and reason about it will occasionally
do one of those jobs to the other's cost.

**Decision:** numbers become digits in code before the model reads them, and every figure in
a reply is checked against a source before it is synthesised.

`vaani/numerals.py` refuses rather than guesses. A run of number words containing anything
it does not know is left exactly as spoken and reported in `unresolved`, which is the input
M1.15's confirmation turn needs. Two adjacent unit words are ambiguous rather than additive,
because Hindi says 55 as one word and "paanch pachaas" is therefore a misrecognition or a
correction, both of which want a human to hear about it.

`vaani/grounding.py` allows a reply the figures the tools returned plus the figures the user
said themselves, and refuses everything else, replacing the reply rather than appending to
it. The check runs per sentence between segmentation and synthesis, which is the only place
it can run: audio cannot be un-said, so a check on a finished reply runs after the wrong
number has been spoken.

**Alternatives considered.** Prompting the model not to invent figures, which is the usual
answer and is a probability rather than a guarantee, on the one number in the system that
sends a person to an office. Allowing derived arithmetic so a reply can say "50000 above the
limit", rejected because allowing derivable numbers allows most numbers and the guardrail
stops being one; the cost is real and is reported as a refusal rate instead of argued away.
Checking every figure regardless of size, rejected because "2. Form bharein" and "do
documents" would refuse almost every reply: the boundary is three digits, which is the same
boundary and the same reasoning `vaani/sentences.py` already uses to tell a list marker from
an amount.

**Consequences.** A legitimate reply that states a true subtraction is refused, and that is
a deliberate false positive whose rate the ablation reports. The validated text is what
reaches the transcript pane as well as the synthesiser, so the screen cannot show a figure
the listener was deliberately not told. Biasing the recogniser toward scheme names (M1.11)
landed in the same pass and carries its own hazard: a Whisper prompt is prepended to the
decoder's context, so near-silence sometimes decodes as the prompt itself, and
`vocabulary.is_echo` treats a transcript made only of bias terms as no speech, because no
real question about a scheme consists solely of scheme names.

## 2026-08-13: Barge-in pauses first and decides second, on two thresholds rather than a lexicon

**Context:** M2.6's remaining half. Barge-in fired on any sustained speech, so "haan",
"achha" and "theek hai" all killed the reply the speaker was agreeing with. Bolna decides
this on a word count over interim transcripts, and it can: Deepgram hands it interims
continuously over a persistent socket. The free stack is chunked transcription, where an
interim costs a whole request over the whole utterance, so counting words during every
reply would mean a provider request every few hundred milliseconds of every turn.

**Decision:** two thresholds on one signal, and a pause that is not yet a cancellation.

Speech during playback past `DEFAULT_COMMIT_MS`, 800ms, interrupts immediately with no
transcript consulted, because nobody backchannels for that long and making a real
interruption wait for a round trip is the barge-in latency we are bounding. Speech past
`DEFAULT_VERIFY_MS`, 200ms, only pauses: the audio is held in the gate's WAIT state, the
utterance is transcribed once after it ends, and `is_backchannel` decides whether the turn
is cancelled or the held audio resumes. That is X-Talk's pause-verify-resume loop, and it
is implementable here only because M2.9 already gave the gate a state that holds audio
rather than dropping it.

The buffered interrupting audio starts the turn it caused. Without it the new turn began
from whatever arrived after the decision, so the agent heard an interruption from its
second word onward and answered a fragment.

**Alternatives considered:** a backchannel lexicon, rejected in the prior-art review
already and worth restating, because Hinglish backchannels are an open set. Interrupting on
duration alone, which is what shipped and cannot tell "haan" from "nahi ruko" at the same
length. Running the recogniser continuously during playback to count words as Bolna does,
rejected on cost: it is the one technique whose price is set by SPEC A4's chunked stack
rather than by design. Committing on the transcript alone with no duration threshold,
rejected because it makes every interruption pay a transcription before the agent stops
talking, which is worse than what we had.

**Consequences.** A verified interruption costs one transcription before the turn is
abandoned, and the agent has already stopped talking by then, so the round trip buys a
resume rather than delaying a reply. Both thresholds are provisional and labelled, set from
the shape of the domain: the recorded corpus sets them, and until then this is a mechanism
whose numbers are a guess even though its structure is not. A recogniser that cannot answer
commits the interruption rather than resuming, because a failed check is not evidence of a
backchannel and resuming over somebody who really did interrupt is the failure the whole
mechanism exists to prevent. `web/index.html` pauses on its own level detector before the
server has decided, so the sound in the room stops without waiting for the round trip, and
the client deliberately decides nothing else: only the server has the transcript that tells
an acknowledgement from a question.

## 2026-08-13: Silero and smart-turn are arms with pinned digests, not upgrades

**Context:** M1.16b ranked the tools worth adding. Two of them are reachable: Silero VAD
for "is this speech at all", replacing a frame-energy threshold of 500 RMS that was written
down as roughly a quiet room and never measured, and smart-turn v3 for "has this turn
ended", against `vaani/completeness.py`'s word-order rule. The backlog deferred smart-turn
on the grounds that it needs torch on a 512MB instance. That is no longer true: the
published `smart-turn-v3.2-cpu.onnx` is 8.68MB and runs on onnxruntime alone.

**Decision:** both are arms behind injected seams, `Endpointer.speech` and
`Endpointer.completion`, with the energy detector and the rule as the defaults. Neither is
selected by a flag, so a bench run constructs both explicitly and neither can be reached by
accident. The model files are fetched by `scripts/fetch_models.py` against a pinned SHA-256
and are not committed; absence is a supported state that runs the baseline arms, and there
is no silent fallback, because a caller that asks for Silero and gets energy would publish
an ablation row for a technique that never ran.

**Two integration details decided rather than discovered.** Silero v5 takes exactly 512
samples at 16kHz and the transport carries 320, and feeding it 320 does not raise: it
returns a confident wrong number. Frames are buffered into windows and the last verdict
stands between them, because reporting zero on the two frames in three that close no window
would make speech stutter at 50Hz and the trailing-silence timer would never complete.
smart-turn is asked once per turn, when trailing silence first reaches the short timeout,
because at up to 100ms an inference fifty times a second is not an option and an earlier
answer is about an utterance that is still arriving.

**Alternatives considered:** replacing the energy detector outright, which is what "adopt
Silero" usually means and which would delete the before that makes the ablation row
meaningful. Committing the weights, rejected because eleven megabytes in a public repo
makes every clone pay for two optional arms. Downloading at import, rejected because a
cold Render container would pay for it on the first turn of the first session, which is
already the slowest turn in the system. Reimplementing the mel filterbank from the formula
and trusting it, rejected in favour of checking it against OpenAI's published
`mel_filters.npz`: it agrees to 1.3e-9, and the version of this that ships broken is the
one where the filterbank is off by a scale factor, the model returns confident numbers from
noise, the suite stays green, and the ablation concludes that neural endpointing does not
help.

**Consequences.** `numpy` is in the dev group rather than only in the `models` extra, so
the filterbank check runs in CI where no model file exists. The model-dependent tests skip
in CI and run locally, which means the arms are exercised where the ablation numbers are
produced and not on every push; that is a gap and it is recorded as one. Neither arm's
accuracy on Hinglish is claimed anywhere: they are wired and verified structurally, and
M4.2's corpus is what decides whether they are better than what they replace. Kokoro is
still not started, for a reason worth writing down: at 82M parameters it needs either torch
or a 310MB ONNX, plus espeak-ng as a system binary for Devanagari phonemisation, and none
of that fits the 512MB instance that M1.8 exists to serve. It stays a bench-only
possibility rather than a deployed one.

## 2026-08-12: Tail control by hedging and transport priority, not by more flushes

**Context:** a tail-control brief proposed five mechanisms for the p95 floor: a hard
350ms time-to-first-token deadline with failover, asynchronous tools with a filler at
300ms, clause-level streaming at four or five words, context-dependent VAD, and a
transport-layer barge-in circuit breaker inside 100ms. Three are improvements, one was
already built, and one would have made the tail worse.

**Decision, per mechanism.**

*Hedge, do not fail over.* A serial failover pays the tail twice to avoid paying it
once: 350ms wasted plus a second time-to-first-token of roughly 200ms is 550ms before
the first token, past the p50 target and 69% of the p95 floor. A hedged second request
fired at the measured p90 makes the worst case the larger of the two rather than their
sum, and costs extra tokens only on the hedged fraction.

*Reject clause-level streaming as architecture, keep it as a measurement.* It saves
roughly 50 to 100ms and costs one TTS request per clause, so three clauses is three
network round trips instead of one. That triples exposure to the variance the p95
floor exists to bound, which makes it a median optimisation that damages the tail. It
also breaks Devanagari prosody across the clause boundary, and a listener hears the
second clause lose its intonation. It becomes a sixth ablation row with its quality
cost reported, because a technique that buys 100ms for three extra requests is worth
publishing either way.

*Adopt context-dependent VAD, as an extension rather than a replacement.* The rule
already built cuts the wait when a partial sounds finished, and it requires three
words because "haan" and "mera" are usually the start of something longer. After a
closed question a one-word answer genuinely is complete, which saves 500ms on
confirmation turns.

*Adopt transport priority.* This identified a live defect: the handler blocks while
sending, and it sends the whole answer as one payload after full synthesis, so an
interrupt can wait roughly 2000ms. Concurrent receive and playback with chunked sends
bounds the worst case at one chunk. That is what makes 100ms plausible, and nothing
else does.

*Asynchronous tool orchestration waits for Dastavez.* The deadline and filler already
exist. The tool stub is in-process and returns in microseconds, so building the
orchestration now means measuring a fixture I wrote, which is the trap Spanlight
recorded when it noticed its own demo tool was written to fail on cue.

**Alternatives considered:** adopting the brief as written, which would have shipped a
serial failover that spends the budget it was meant to protect, and a clause-level
flush that trades the p95 target for a median gain. Adopting nothing, which would have
missed the transport defect and the confirmation-turn case, both of which are real.

**Consequences, and this is the part that governs the order of work.** Every number in
the brief is intuited: 350ms, 300ms, 100ms, four or five words. This portfolio has
twice paid for that. ShipGate gated on 2 points against a judge whose measured noise
floor was 20, and Spanlight shipped a threshold that fired on a pattern written down
as healthy. So the mechanisms get built with their thresholds as named,
labelled-provisional constants, and the numbers are set from measured distributions
once M0.1 exists. M2.4 stays measured rather than targeted for the same reason, and
that is the answer to the 100ms figure: it is a budget we verify, not a target we
assert.

## 2026-08-12: The target is p50 under 500ms and p95 under 800ms, and filler does not count

**Context:** the goal was "sub-1000ms time to first audio", a single number with no
distribution attached. The target is now p50 under 500ms with a p95 reliability
floor under 800ms, which is a different problem rather than a tighter version of the
same one.

**Decision:** SPEC's budget table gains a p95 column, and the p95 is what decides
the technique list. A median is bought by optimisation: remove a wait, overlap two
stages, skip a request. A tail is not, because the tail is a provider having a bad
second and no pipeline design makes a free-tier endpoint answer faster when it is
queueing. So the p95 is reached by bounding what can go wrong: per-stage deadlines,
a local synthesiser whose tail is a CPU rather than a network, and something audible
guaranteed by a deadline at 600ms.

Three of the four optimised terms now have no network on the critical path at all,
which is the p95 column deciding the design. It is also the argument for M1.8's
local synthesis, which looked marginal on the median and is the fattest remaining
tail.

**The measurement rule matters more than the mechanism.** A filler acknowledgement
is audio, and counting it as time to first audio would let any system hit any target
by learning to say "achha" quickly. `TurnClock` keeps two numbers, time to first
audio of any kind and time to first audio of the answer, and every target is judged
against the second. A configuration that met p95 by talking over the gap publishes
its filler rate beside the number. The tests mutate all three of those guards,
because this is the exact shape of dishonesty the project was built to avoid and it
would be trivially easy to commit here.

**Alternatives considered:** treating the deadline as a cancellation, so a late
answer is abandoned and only the filler is spoken. Rejected because it trades a slow
answer for none, and a listener who hears an acknowledgement and then nothing has
been failed worse than one who waits. Cancelling the pending first chunk when the
deadline passes, rejected for the same reason at a smaller scale: it closes the
generator mid-flight and throws away work already done, which is the opposite of
what a deadline is for.

**Consequences:** 150ms of headroom separates the p95 estimate of 650ms from the
800ms floor, and one retained network round trip would spend it. The budget is
therefore a claim to be falsified rather than a plan, and if measurement puts p95
above 800ms the finding is that a free-tier cascade cannot hold a tail that tight.
That gets reported as the result. `FIRST_AUDIO_P50_MS` and `FIRST_AUDIO_P95_MS` live
in `vaani/budget.py` so a bench script and a CI assertion read the same numbers, and
a test pins them to the published figures so moving a target is a visible change
rather than an edit in one file.

## 2026-08-12: A fresh partial becomes the final transcript, and the tail decides

**Context:** SPEC's budget allows 100ms for the STT tail after the endpoint, on the
reasoning that streaming means the final partial already exists when the endpoint
fires. `ChunkedStt` did the opposite: it always transcribed the whole utterance
again after the frame stream ended. On the free stack that is a full request over
the whole utterance, roughly 400ms on the critical path, to arrive at a string it
usually already had. Committed code contradicting a committed number.

**Decision:** the final reuses the last partial when the audio not yet transcribed
is no longer than `reuse_final_within_ms`, defaulting to 700ms, which is the
trailing silence that ends a turn. The frame stream ends at the endpoint, so that
tail is by construction the silence which caused the endpoint: speech stopped
before it and nothing was said during it, so the last partial contains every word
spoken. A longer tail means partials fell behind or failed, the words at the end
were never transcribed, and the request is real work rather than a duplicate.

The partial interval also drops from 600ms to 400ms. A fresher last partial is what
makes reuse fire, and on this stack each partial is a request, so the interval is
the dial between provider spend and how often the tail term is paid.

**Alternatives considered:** always transcribing the final, which is what shipped
and cannot meet the budget. Always reusing the last partial, rejected because a
stack whose partials failed would silently answer a question it never heard the end
of. Taking a partial at the endpoint instead of on a timer, which is the right
answer and needs the endpointer to signal the recogniser; that coupling is a wiring
change, and `reuse_final_within_ms` is the seam it would use.

**Consequences:** the reused final carries the same `index` as the partial it
reuses, and two events sharing one index is the honest encoding, since it says no
further request was made. `vaani.stt.final_reused` is on the stage span so the
waterfall can separate turns that paid the tail from turns that did not, which
matters because the two are different latencies and averaging them hides the
technique. The correctness risk is real and stated: if the endpoint fires while the
user is still speaking, reuse propagates the truncation instead of catching it, so
this technique and semantic endpointing share a failure mode and the ablation has to
report their false-endpoint rate together rather than separately.

## 2026-08-12: A session is one turn, and stage spans are not tool spans

**Context:** wiring SPEC's span tree turned up two choices that look like details
and are not.

**Decision:** a Spanlight session wraps one turn rather than one socket, with a
`turn` span inside it carrying the attributes SPEC names. And stage spans are
created from `spanlight.get_tracer()` directly, not `spanlight.tool_span`.

**Alternatives considered.** A session per socket, which is what SPEC's tree
implies, rejected because detector state in Spanlight is per session: a session
holding a whole conversation reads the repeated tool calls of three separate
questions as a loop and fires on a healthy dialogue. ShipGate hit the same thing
and moved to per-item sessions. The cost is that a multi-turn conversation is
several traces rather than one, and cross-turn analysis becomes a query.

`tool_span` for the stages, which is the obvious helper and is wrong here: it names
the span `tool <name>` and stamps the tool attributes on it, so five pipeline
stages a turn would look like repeated tool calls to the loop detector and to the
silent-tool-failure detector. That is manufacturing detections out of a healthy
pipeline, which is the failure mode Spanlight measured at 14.3% and 28.6% of
healthy sessions before its rules were rewritten.

**Consequences:** `record_exception` and `set_status_on_exception` are both off on
every stage span. Their defaults attach `exception.message` and a full stack trace,
and this pipeline's exceptions carry provider error bodies that can quote a
transcript back. The canary test for that passed against the leak until its fixture
was given a genuinely recording tracer: with no endpoint configured every span is a
`NonRecordingSpan` and nothing is written to it either way, so both settings
behaved identically.

`playback.first_audio` should end when the browser reports playback started, and
the browser does not report yet, so it closes immediately with
`vaani.playback.reported=false`. Queryable and honest, where a span that silently
never ends would be neither.

## 2026-08-12: A stream is retried only until its first event

**Context:** `ChatClient.complete` is wrapped in `retry_with_backoff`, and the
streamed path cannot be. A decorator sees one call succeed or fail, while a
stream stops being repeatable the moment a token is handed downstream: by then
the caller may already have synthesised it and put it in somebody's ear.
Reconnecting would either repeat the opening words or splice two different
replies into one sentence.

**Decision:** streaming lives in the chassis `llm/` rather than in `vaani/`, so
Tollgate inherits it and Vaani does not duplicate the throttle gate and the
retry loop. `ChatClient.stream` runs its own loop instead of the decorator:
before the first event a failure is transient and retried, and after it the error
propagates and the tokens already delivered stay delivered. The 429 gate stays
inside the loop for the reason the unstreamed path records. `backoff_seconds` in
`llm/retry.py` mirrors tenacity's curve so the two paths wait the same way.

Providers assemble tool-call fragments rather than forwarding them. Arguments
arrive a few characters at a time, the last frame is often a closing brace, and
every caller left to reassemble them would get the indexing wrong differently.
`stream_options: {"include_usage": true}` is requested, because a streamed
response omits usage otherwise and the ablation would compare a measured
unstreamed baseline against a column with no tokens and no cost in it.

**Alternatives considered:** a Vaani-local streaming client, rejected because it
would reimplement the throttle, the retry, and the retry-after parsing, which is
where the double-counting and invisible-retry bugs came from in Spanlight.
Retrying mid-stream and discarding the tokens already sent, rejected because they
are not recoverable: playback is downstream and audio cannot be un-said.
Buffering the stream until it completes so it stays retryable, which is the
unstreamed path with extra steps and gives up the entire point.

**Consequences:** the streamed path carries a weaker guarantee than the
unstreamed one, and that difference is a row the ablation write-up owes the
reader rather than a detail. A mid-stream failure surfaces as a half-spoken reply
until M3.2 gives it a degradation path. `llm/client.py` and
`llm/providers/base.py` change for all eleven forks, which is the cost the
chassis placement buys, so `complete` is untouched and the streaming tests
mutation-checked in both directions. `ChatMessage` gained optional `tool_calls`
and `tool_call_id`, serialised with `exclude_none`, so a provider that has never
seen those fields receives the payload it did before.

One test was green for the wrong reason and worth recording. The check that a
4xx body is read before the status is mapped passed with the `aread` call
deleted, because `httpx.MockTransport` with a bytes body hands back a response
that is already buffered. It only became a real test once the fake body was an
async iterator, which is what a live connection gives you.

## 2026-08-12: The tool stub answers indicatively and refuses rather than guessing

**Context:** M1.3 needs the Dastavez tools to exist before Dastavez does. A stub
that returns plausible eligibility verdicts is trivial to write, and that is the
problem: this pipeline speaks its answers aloud to somebody asking whether they
qualify for a welfare payment, and a threshold invented for a unit test is
indistinguishable in the ear from a sourced one.

**Decision:** `vaani/tools.py` holds five schemes with one or two thresholds
each. Every result carries `indicative=True`, and a scheme with no thresholds
recorded raises rather than returning eligible, since `all([])` is true and "we
have nothing to check" is not "you qualify". An unknown scheme id and an unknown
tool name both raise instead of returning an empty result, because an empty result
teaches the model the call succeeded. Arguments are validated at this boundary
with pydantic and `extra="forbid"`, and a rejection names the field and the rule
but never the value, which is an applicant's income.

The verdict is computed from the comparisons rather than from the sentences they
produce. The first version derived it by testing whether the word "within"
appeared in prose formatted three lines earlier, so rewording a message would
have flipped eligibility answers with the whole suite still green.

**Alternatives considered:** richer fixture data covering thirty schemes,
rejected because volume is what makes a stub start looking like a source of
truth, and five exercise every branch the pipeline has. A `confidence` derived
from how many rules matched, rejected because it would be read as a probability;
it is a constant and a test pins it as one. Returning structured errors instead
of raising, which would let the model correct itself inside the turn, rejected
for now because retrying a tool call in a loop is what Spanlight's loop detector
exists to catch, and the turn-level handling belongs to M3.5.

**Consequences:** the module is `vaani/tools.py` rather than the `vaani/tools/`
package SPEC's architecture table names, matching the flat layout the rest of
`vaani/` already uses. The README has to state that eligibility answers are
indicative and that Dastavez replaces the data, since the demo will be spoken to
by people who did not read this file. The threshold matrix in the tests covers
each limit exactly and one either side: a matrix of round numbers passed an
off-by-one comparison, because the edge is the only place `<=` and `<` differ.

## 2026-08-12: The chassis bootstrap is deleted, not wrapped

**Context:** `pyproject.toml` declared Spanlight from the first commit, and
`app/main.py` was still calling the inherited `app/otel_bootstrap.py`. Nothing
called `spanlight.init()`. The bootstrap carried both faults Spanlight recorded
and fixed in August: it passed the endpoint straight to `OTLPSpanExporter`, which
appends nothing, so every span went to a URL that does not accept spans, and it
never percent-decoded `OTEL_EXPORTER_OTLP_HEADERS`, so Grafana's
`Authorization=Basic%20...` went out literally and earned a 401 that reads like a
bad credential. This repo was the fourth copy.

**Decision:** delete the module and call `spanlight.init()` from `create_app`.
Vaani also stops declaring opentelemetry at all, since nothing here imports it:
Spanlight declares the SDK and the OTLP exporter, and the FastAPI instrumentation
the chassis listed was imported nowhere.

**Alternatives considered:** keeping both and having `setup_otel` delegate,
rejected for the reason Spanlight rejected it. OpenTelemetry ignores a second
`set_tracer_provider` and only logs about it, so whichever ran first would win
silently and the loser would export nothing while appearing configured. Fixing
the two bugs in place, rejected because it makes a fifth copy of code that now has
a maintained home.

**Consequences:** the deployed backend has been exporting nothing since M0.5, so
no trace from before this commit exists and the M1 demo checkpoint has to be
re-run. HTTP server spans go away with `FastAPIInstrumentor`, which is no loss:
the traffic is a WebSocket and SPEC's span tree is session, turn, then stage. The
deletion is held by a test rather than a comment, because a fork is how this came
back the last three times.

## 2026-08-12: A full stop after a number ends a sentence when the number is long

**Context:** sentence segmentation suppressed every full stop that followed a
digit, to protect decimals ("2.5 hectare"), abbreviated amounts ("Rs. 6000") and
numbered list markers ("2. Form bharein"), all of which occur in almost every
eligibility answer. The rule was too broad in two ways, and both cost the thing
this milestone exists for. `Aapki income limit hai 300000.` produced no sentence
at all, so synthesis waited for the whole reply. `Aapki limit hai 300000. Aap
eligible hain.` merged into one sentence, so the first one never flushed early.
Devanagari hid both, because `है।` terminates on a danda and never reaches the
rule, which means the failure was confined to Hinglish: exactly the case SPEC S2
is about.

**Decision:** a full stop following a run of `MIN_AMOUNT_DIGITS` or more digits
ends a sentence. Below that it is a list marker and does not. A full stop between
two digits stays a decimal regardless of length. The threshold is 3, set from the
shape of the domain rather than measured: markers are one or two digits, while
amounts, years and pincodes are three or more.

Separately, a full stop after a digit at the very end of the buffer is
undecidable mid-stream, since "2." can still become "2.5" when the next token
arrives. `split` takes a `final` flag, so the decision waits while tokens are
still coming and resolves as a full stop once the reply has ended.

**Alternatives considered:** keeping the flat rule and accepting that answers
ending in an amount do not flush early, rejected because that is the commonest
shape of answer in the domain and the loss is invisible: the sentences all arrive,
just later, so nothing fails and the ablation quietly measures a smaller win.
Splitting on the digit-final full stop unconditionally, rejected because in a
token stream it cuts decimals in half and synthesis says "two point" and then
pauses for a round trip before "five". Requiring a following space and capital
letter, rejected because Devanagari has no capitals and Hinglish replies switch
script mid-sentence.

**Consequences:** the threshold is a heuristic sitting in the path of the
project's headline measurement, so it is a named constant with its reasoning
beside it rather than a condition. It is bracketed by tests from both sides:
raising it fails the amount cases, lowering it to 1 fails the list marker. A
number written as "2.5 lakh" and a sentence genuinely ending in a two-digit
number, such as "aapki umar 45.", still merge forward, which is a real remaining
gap and is cheap to revisit if the eval set turns up examples.

## 2026-08-12: The interruption flag is named for the turn it describes

**Context:** the turn state machine recorded barge-in as `interrupted`, set
during `begin()` from whether the machine was thinking or speaking. The value is
correct at the instant it is written, and then it persists. Turn two is born from
a barge-in and turn two is not itself interrupted, so for the whole life of turn
two the flag reads true about the wrong turn. M1.6 puts this on
`vaani.turn.interrupted` at the moment a turn's span closes, so the ambiguity
would have shipped as a wrong span attribute on every turn following the first
interruption of a session.

**Decision:** the field is `interrupted_previous`, and it reports whether the
turn being displaced was cut off. That is what SPEC S4 asks to be recorded, and
it is unambiguous at the only moment anything reads it. The class is `TurnState`
rather than `Turn`, because `vaani/turn.py` already has a `Turn` and two classes
of the same name in one package is a mistake waiting for a tired afternoon.

**Alternatives considered:** keeping the name and documenting when it is valid,
rejected because a flag that is true about a different object than its name
suggests is read wrongly by whoever did not write it. Recording interrupted
generations in a set, which answers the question for any past turn rather than
only the last one, rejected as unbounded state for a question nothing currently
asks; the span closes at rollover and that is the only reader.

**Consequences:** M1.6 reads the flag while closing the displaced turn's span,
not while opening the new one, and the tests say so. If a later milestone needs
the interrupted status of an arbitrary past turn, the set is the change to make
and it should be a deliberate one rather than a widened flag.

## 2026-08-11: The frontend ships on GitHub Pages, not Vercel

**Context:** SPEC and BACKLOG both name Vercel for the browser client. The client
turned out to be two static files, `web/index.html` and `web/capture-worklet.js`,
with no build step and no server-side anything.

**Decision:** GitHub Pages. Recorded here because the backlog states the
deviation is recorded in DECISIONS, and it was not.

**Alternatives considered:** Vercel as planned, which is a better fit the moment
the client needs a build, an environment variable, or a rewrite rule, and none of
those exist. It also means another free-tier account to create, hold credentials
for, and record in QUOTAS.md.

**Consequences:** one fewer provider in the quota table and one fewer account
that can expire. Pages serves from the same repository that already deploys the
backend, so the frontend has no separate deploy step to forget. If the client
grows a build, this decision is the one to revisit, and SPEC's architecture table
still says Vercel until it is corrected.

## 2026-08-11: WebSocket for the transport, not HTTP polling or WebRTC

**Context:** the browser has to send a continuous stream of audio frames and
receive audio, transcript text, and control messages back, in real time, in
both directions, for the life of one conversation. Never formally written down
before now, though implicit from `app/routers/voice.py`'s first commit; M6.4
names the transport choice explicitly as a decision this file has to carry.

**Decision:** one WebSocket per session, framed messages in both directions
(binary audio frames, JSON control and text), matching the shape
`vaani/protocol.py` and `web/capture-worklet.js` already implement.

**Alternatives considered:** WebRTC, which is the standard answer for
browser real-time audio and the one a production voice product would very
likely use, rejected here on cost rather than merit: it needs a signalling
server, STUN and likely TURN infrastructure for anything past a local network,
and audio codec negotiation this project has no use for, since the audio
never needs to survive a lossy, jittery public network path the way a video
call does; it goes to one backend process over a normal TCP connection.
Adopting it would have spent a meaningful slice of this project's own time
budget on transport plumbing orthogonal to what it measures. HTTP long-polling
or repeated small requests, rejected outright: the barge-in latency this
project measures in milliseconds would be dominated by request overhead
before any pipeline stage got a chance to matter.

**Consequences:** the whole session lives on one connection, which is why
`app/routers/voice.py`'s single-session lock (M3.8) is a simple
process-wide `asyncio.Lock` rather than anything more elaborate: there is
exactly one thing to serialise access to. A real product serving more than
one caller at a time would need WebRTC's own infrastructure or a different
scaling story; this decision is scoped to what a measurement project on a
single free-tier instance needs, stated as a scope limit rather than implied
as a production architecture.

## 2026-08-11: Cascaded pipeline, not a speech-to-speech model

**Context:** the two live architectural options for a real-time voice agent are
a cascaded pipeline, STT then an LLM then TTS as separate stages, and a
speech-to-speech model that reasons and generates directly in the audio
domain. Non-goal 2 in SPEC already rules S2S out; this entry is the reasoning
behind that line, requested by name in M6.4 rather than left implicit in a
non-goals list nobody reads as a decision.

**Decision:** cascaded. Every stage in this project, `vaani/stt.py`,
`vaani/llm_turn.py`, `vaani/tts.py`, is a separate, independently swappable
component behind its own interface, which is also what M4's whole waterfall
comparison depends on: SPEC A4's chunked-versus-streaming difference and
M4.4's Groq-versus-Sarvam comparison both live at seams a cascade has and an
S2S model does not.

**Alternatives considered:** a genuine speech-to-speech model (Moshi-shaped,
or a hosted S2S API), rejected for three reasons that compound rather than
stand alone. It needs a GPU this project's free-tier budget does not have, or
a hosted S2S API this project's quota table does not carry credits for
either. It collapses exactly the seams this project's contribution depends
on measuring: there is no separate STT stage to swap for a streaming one, no
separate TTS stage to fail over, nothing resembling SPEC A4's comparison at
all. And this project's specific eligibility-checking task needs a real tool
call against real fixture data with a real refusal path when the model
cannot verify a figure (`vaani/tools.py`, `vaani/grounding.py`), which is a
solved, ordinary problem for a text-reasoning LLM in a cascade and an open
research problem for an audio-native model to do reliably.

**Consequences:** the entire depth chapter, M5's ablation, is only a
coherent thing to build because the pipeline is cascaded; an S2S model would
have nothing resembling separable stages to ablate. The Depth Ladder's own
item 3, "a speech-to-speech comparison row on the same corpus," is what
keeps this decision from reading as never having considered the
alternative: it is deferred, not dismissed, and M2.14's ConversationEngine
seam (also deferred, this session, for time rather than merit) is what
would let a future S2S arm be measured behind the same interface without
replatforming the rest of this project.

## 2026-08-18: The rest of SPEC's A1 through A10, where they are not their own entry

**Context:** M6.4 asks this file to carry A1 through A10. Four already have entries
whose title does not literally quote "SPEC A4" or similar: chunked-versus-streaming
STT (A4, the STT interval and provider-fallback entries), the Sarvam credit
discipline (A5, QUOTAS.md and M4.4's own entry), the synthesised corpus (A8, M4.2's
entry), and the single-session lock (A10, M3.6 and M3.8's entries). Transport (A1)
and cascaded-versus-S2S, adjacent to A2, are the two entries directly above this
one. What remains, A2, A3, A6, A7, A9, is either settled by inheritance rather than
choice, or already visible in tracked, non-DECISIONS files, and each is recorded
here rather than left for a reader to assume was never considered.

**A2, Tollgate and Dastavez do not exist yet.** Not a decision this project made;
a fact about the portfolio's build order (this project is third of eleven,
memory recorded in `read-sibling-repo-learnings-first`). `vaani/tools.py`'s
fixture-backed schemes and `llm/`'s chassis client are what stand in for them.

**A3, Python not Node.** Inherited from the chassis template this repo forked
from (`M0.2`, "chassis fork cleaned"), not chosen fresh; Spanlight's own
per-stage instrumentation is Python, and matching it rather than bridging two
languages for one process is why this was never genuinely reconsidered.

**A6, Kannada supported but unscored.** Enforced in the eval set's own header
per the acceptance text, not in code; `docs/prior-art.md`'s HiACC note
(M4.11) is the closest thing to a decision record for the boundary, since
the exclusion is a labelling-integrity choice rather than an engineering one:
this project's own DECISIONS entries on measured-not-guessed thresholds
(the hedge delay, the STT interval) are the same discipline applied here to
who is allowed to adjudicate a label rather than to a number.

**A7, a comparison states its clock or is not made.** Applied, not merely
stated: the X-Talk comparison (M4.14, `docs/prior-art.md`) publishes both
clocks side by side rather than picking the flattering one, and M6.2's own
task text does the same for the uncited industry figure it declines to
repeat.

**A9, Render free spins down, cold starts flagged.** `web/index.html`'s own
cold-start message ("Render's free instance spins down when idle. This wait
is a cold start.") is the acceptance criterion met in the client rather than
in a benchmark script; M4.3's waterfall does not yet exclude a cold-started
run from its own statistics, which is the gap M6.6's clean-browser
verification is positioned to catch before publication.

**Decision:** record all five here rather than open five near-empty entries for
facts with no real alternative that was weighed, which would read as decisions
where none were made.

**Consequences:** the five above point at the file, entry, or fixture that is
each one's actual evidence, so M6.4's own acceptance, "A1 through A10... in this
file," is satisfiable by a reader following those pointers rather than by
duplicating their content here.

## 2026-08-18: The unstreamed baseline's first clock skipped the wait it was supposed to pay

**Context:** M5.2's ablation harness measures the streamed pipeline against the M0
unstreamed baseline (`vaani/turn.py`). The first version of the unstreamed arm
called `Turn.run(pcm)` on the corpus utterance's whole buffer immediately and timed
that call alone. The result: the "slow, naive baseline" arm measured faster than
the streamed arm it exists to be measured against, which is backwards. Investigated
rather than reported, because a baseline built to be the floor coming in under the
optimised path is a bug in the measurement, not a finding about the pipeline.

The cause is exactly the shape CLAUDE.md's own history warns about: a span measures
its own extent, so where you start it is the measurement. `bench/stages.md` defines
time to first audio from the last frame of user speech, which for the streamed arm
includes the real, paced, real-time wait for the endpoint's trailing silence to
elapse, several hundred milliseconds to a full second depending on aggressiveness.
The unstreamed arm's first version had no frame pacing and no endpointer at all, so
its clock started at "whenever this function was called" with that wait already
absent, structurally, not measured away.

**Decision:** the unstreamed arm now paces frames through a real `Endpointer` the
same way the streamed arm's own corpus playback does, and backdates its clock to
the endpoint-fire time minus the trailing silence, `endpointer.silence_ms`, the
same computation `vaani/session.py` already uses for the streamed arms' own clock.
Both arms now start counting from the same event.

**Alternatives considered:** subtracting a flat, assumed trailing-silence constant
from the unstreamed arm's own wall-clock time after the fact, rejected because the
actual wait depends on the endpointer's own state (semantic completeness can end it
early), and a flat correction would be exactly the kind of assumed-not-measured
number this project's own DECISIONS entries keep finding and fixing elsewhere
tonight.

**Consequences:** after the fix, a small (n=2) run found the streamed arm faster by
376ms, the predicted direction; a separate small (n=3) run across all three arms
found it slower by 2316ms. Both are real, both used the corrected clock, and they
disagree with each other, which is itself the finding: at n this small, ordinary
per-call latency variance dominates whatever the true technique effect is, and
`ablation/hypothesis.md`'s own n=20 is not a formality. M5.2 stays open rather than
publishing either small run as the number.

<!-- Add entries above this line. -->
