# Stage boundaries

What every stage span starts and ends at, written before anything is measured.

This file exists because of a specific failure in the previous project. Its field
study published a session latency of 1184ms while the provider's own median was
559ms, because the session span was opened before a concurrency semaphore was
taken, so half of it was queueing reported as work. Nobody noticed until a single
trace was opened by hand, after the aggregates had already looked reasonable. A
span measures its own extent, so choosing where it starts is choosing what gets
measured, and "STT time" is ambiguous by roughly 200ms depending on whether you
count from the first audio frame, from endpoint detection, or from request
dispatch.

So the definitions come first and the instrumentation is written to them. M4.1
hashes this file before M4.2 measures anything, and the commit order is visible in
the log.

## The rule these all follow

A stage span covers work this process is doing or waiting on for that stage
alone. It never covers:

- waiting for an admission point: a semaphore, a pool checkout, a rate-limit gate
- another stage's work, even when the two overlap in wall-clock time
- a retry loop's total, without an attempt span inside it to show the attempts

Where a wait is unavoidable inside a span, it gets its own child span or event, so
waiting can be told from working.

## Stages

### `vad.endpoint`

**Starts** on the first audio frame whose RMS clears the threshold, which is the
first frame attributed to this turn's speech. Not on socket open, and not on the
first frame received: a user who takes three seconds to begin talking would
otherwise have three seconds of silence recorded as detection work.

**Ends** the instant the endpointer returns true, before any transcription request
is built.

**Excludes** leading silence, and the trailing silence itself is inside the span
because it is the detector's own cost, not a wait on anything external. That is a
choice rather than an obvious truth: it means `vad.endpoint` duration is roughly
speech plus `trailing_silence_ms`, so the number moves when the aggressiveness
knob moves. That is the intent, since the knob's whole purpose is trading dead air
against cutting people off, and a span that excluded the timeout would show the
knob having no effect.

`vaani.vad.speech_ms`, `vaani.vad.trailing_silence_ms`, `vaani.vad.aggressiveness`.

### `stt.stream`

**Starts** when the first audio frame is handed to the recogniser, which on the
chunked stack is when the first partial request is dispatched and on a streaming
stack is when the socket is opened.

**Ends** when the final transcript is in hand.

**Excludes** the endpoint wait, which belongs to `vad.endpoint`. The two overlap in
wall-clock time on a streaming stack and must not overlap in attribution, or the
same milliseconds appear in both columns and the waterfall adds up to more than the
turn.

Each provider request is its own child `stt.request` span. Without them a stack
that made four requests and one that made one produce spans differing only in
duration, which is the retry-invisibility failure the previous project recorded:
wrap the call and you cannot see the attempts inside it. On the chunked stack the
count of these children is the cost of a partial, which SPEC A4 says must not be
glossed over.

`gen_ai.system`, `vaani.stt.partials`, `vaani.stt.final_chars`,
`vaani.stt.streaming`.

### `llm.generate`

**Starts** at request dispatch for the first round of the turn.

**Ends** when the model stops producing tokens for the last round, including every
tool round trip. One span per turn, not per round, because the caller waited for
all of them.

**Includes** the tool round trips as child `tool.*` spans, and each provider
attempt as a child attempt span. The parent says what the call cost the listener;
the children say what it took. A 429 that asked for forty seconds would otherwise
make a rate-limited call look fast.

Time to first token is an event on the span, not a second span:
`vaani.llm.first_token`. It is the number a listener experiences, and it is not the
span's duration.

`gen_ai.*`, `spanlight.cost_usd_equivalent`, `vaani.llm.rounds`.

### `tts.synthesize`

**Starts** when a sentence is handed to the synthesiser. One span per sentence, not
per turn, because first-sentence flush means sentence one and sentence three are
separate requests with separate latencies, and averaging them into a turn-level
span hides the thing M1.4 was built to buy.

**Ends** when the last audio chunk for that sentence has been yielded.

Time to first chunk is an event, `vaani.tts.first_chunk`, for the same reason as
the model span: the duration is how long synthesis took, and the event is when the
listener could have started hearing it.

`vaani.tts.voice`, `vaani.tts.sentence_index`, `vaani.tts.chars`,
`vaani.tts.chunks`.

### `playback.first_audio`

**Starts** when the first audio chunk of a turn is written to the socket.

**Ends** when the browser reports it has begun playing. That report is a client
message, so this span is the only one whose end depends on something outside the
process, and a client that never reports leaves it unclosed. It is therefore
bounded and marked `vaani.playback.reported=false` when the bound is hit, because
a span that silently never ends is worse than a short one that admits it.

`vaani.playback.queued_ms`, `vaani.playback.reported`.

## Time to first audio, which is the headline and needs two numbers

**Starts** at the last frame of user speech. Not at endpoint-detected, not at
request-received, not at socket receive. The endpoint wait is inside the number,
which is between 250 and 700ms that most published figures exclude, and it is the
part a listener actually experiences as the agent being slow to react.

**Ends** at the first audio chunk of the turn being written to the socket.

Three numbers are recorded and all three are published:

- `first_audio_ms`, the first audio of any kind, filler included.
- `first_answer_audio_ms`, when the answer's first chunk was written to the socket.
- `first_answer_heard_ms`, when that chunk could reach the ear, which is the send
  time plus whatever was still queued to play in front of it.

**The headline target is the third one.** The second looks like the right number and
is not, and the reason is the filler. `speak_within` yields the whole filler before
the answer's first chunk, audio leaves the process far faster than real time, and the
committed clips are between 1.8 and 3.5 seconds of playout. So on a turn where the
filler fired, `first_answer_audio_ms` records a moment at which the listener is still
hearing "ek minute". Judging the target on it flatters every such turn, and the filler
fires on most of them, because the deadline is shorter than the endpoint wait.

That is the same error the clock choice above exists to prevent, one stage further
down: a measurement taken where the code was convenient rather than where the listener
is. It is recorded here rather than only fixed, because the fix is a definition.

A transport that cannot estimate playout falls back to the send time, which is known
to be optimistic and is reported as the send time rather than compared as though it
were the same number.

A filler acknowledgement is audio, so a system counting it as the answer could hit any
target by learning to say "achha" quickly. A configuration that met p95 with filler
publishes its filler rate beside the number, and a turn covered by filler is a turn
the answer was late in.

The clock is backdated rather than started at the endpoint: `silence_ms` has already
elapsed by the time the endpointer fires, so `TurnClock` subtracts it. An earlier
version read that value after the endpointer had been reset, so the backdating was
always zero and the flattering clock shipped behind a comment saying it had not.
Where you start the measurement is the measurement, and so is when you read it.

## Barge-in latency, which is now three numbers rather than one

M2.4 measures this and it got harder to define, because barge-in stopped being one
event. Speech over the agent is a hypothesis first and a decision second, so there
are three boundaries and reporting only the friendliest of them would be the same
dishonesty as starting the first-audio clock after endpointing.

**Starts**, for all three: the first frame of the interrupting speech, meaning the
first frame that later turns out to have been part of it. Determined after the fact
from the frame log, not from when detection fired, for the same reason the
first-audio clock starts at the last frame of speech rather than at the endpoint.

- **`bargein.paused_ms`** ends when audio stops reaching the listener. That is the
  earlier of the client pausing its own playback on its level detector and the
  server's PAUSE arriving. This is the number a listener perceives as the agent
  shutting up, and it is the one a 100ms budget is about.
- **`bargein.committed_ms`** ends when the turn is actually abandoned: the
  generation moves and in-flight generation and synthesis are cancelled. On the
  duration path this is 800ms of sustained speech by construction. On the verified
  path it includes a transcription round trip.
- **`bargein.resumed_ms`** ends when held audio resumes after a backchannel. It only
  exists on turns that were not interruptions at all, and it is the cost the
  verification arm charges for not cutting somebody off who said "haan".

**Excludes** nothing on the grounds of being somebody else's fault. The
transcription in the verified path is inside `committed_ms` even though it is a
provider's latency, because the user is waiting through it either way.

Reporting only `paused_ms` would be the flattering number: audio stops quickly and
the turn is still alive behind it. Reporting only `committed_ms` would understate
what the listener perceives. Both are published, with the split between duration-path
and verified-path turns stated, because they are different distributions and averaging
them hides the technique.

## The playout estimate is not a measurement

`TurnTaking._playing_until` is an estimate of when queued audio finishes playing,
advanced by each chunk's own duration rather than by when it was sent, because audio
is handed to the transport far faster than real time. It exists to decide whether to
hold a chunk, and it is derived: byte count divided by the synthesiser's declared
encoded rate.

It is **not** a latency measurement and must not appear in the waterfall. A provider
that cannot declare its rate returns zero, and then the estimate does not advance at
all, which is the honest failure. `vaani.playback.queued_ms` on the span is a
different quantity from this one and the two are easy to confuse, which is how the
constant behind this estimate went unchecked at five times the correct value.

## What the turn span is

`turn` wraps all of the above for one utterance. Its duration is the number a
listener would call "how long did that take", from the first frame of their speech
to playback starting.

It deliberately does not start at socket open or at session start. A session holds
many turns and a user pausing between questions is not latency.

`vaani.turn.index`, `vaani.turn.interrupted`.

## The sum will not match

Stage durations do not add up to the turn duration, and that is the point rather
than an error to reconcile. The stages overlap: STT is still finishing while the
model has started on a speculative prefill, and synthesis of sentence one runs
while the model writes sentence three. The unstreamed baseline is the configuration
where they nearly do add up, which is exactly why it is kept.

Any report that presents stage times as a partition of the turn is wrong, and the
waterfall renders them as overlapping bars on a shared timeline for that reason.
