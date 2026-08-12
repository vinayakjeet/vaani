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
