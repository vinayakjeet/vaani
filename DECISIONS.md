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
