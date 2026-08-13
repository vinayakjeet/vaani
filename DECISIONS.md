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

<!-- Add entries above this line. -->
