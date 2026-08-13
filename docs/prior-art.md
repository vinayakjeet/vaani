# Prior art, and what Vaani takes from it

Read against the TrueState voice-agent blueprint (July 2026), LiveKit Agents, Pipecat,
TEN Framework, Bolna, Vocode, Dograh, and the turn-taking literature.

One thing first, because it decides everything below. **Vaani is not competing with these
systems and should not try.** LiveKit and Pipecat are transport and orchestration
infrastructure with teams behind them; TEN is a multi-language extension graph; Bolna and
Dograh are self-hostable platforms with real telephony. Vaani is a browser microphone, a
free tier, and no GPU, and SPEC's non-goals exclude telephony, speech-to-speech,
self-hosted inference and cross-session memory on purpose.

What Vaani has that none of them publish is **a measured ablation with an honest clock**.
Every one of these projects ships mechanisms; none of them publishes which technique bought
which milliseconds, on what corpus, with variance, including the techniques that bought
nothing. That is the axis where being better is available, and it is the only one worth
contesting.

## Confirmed by prior art, already built

Recorded because independent agreement is evidence, and because two of these we arrived at
by measurement rather than by reading.

| Mechanism | Where they say it | Where we do it |
|---|---|---|
| Sentence-level TTS streaming | Blueprint W3-4 | `vaani/tts.py` |
| Partials feeding a semantic end-of-turn decision, 700ms down to 250-400ms | Blueprint 3.1 signal 3 | `vaani/completeness.py`, 700 to 200 |
| Pre-synthesised filler, "cached audio, ~0ms" | Blueprint 3.2 layer 4 | `vaani/assets/filler-hi.mp3` |
| Every vendor behind a swappable interface | Blueprint 5.2, closing truth 4 | `SttProvider`, `TtsProvider`, `Provider` |
| Per-stage OTel spans with per-turn trace ids | Blueprint 5.2, "you cannot fix latency you cannot see" | `vaani/spans.py`, `bench/stages.md` |
| Structured tool query rather than vector search | Blueprint 3.2 layer 1 | `vaani/tools.py` |
| Health-checked fallback chains per component | Blueprint 5.2 | `RecoveringStt`, `FailingOverTts`, `stream_hedged` |
| Eval harness gating CI, guardrail violations must be zero | Blueprint W9-10 | M4.7 |

The filler one is worth dwelling on. We found it by measuring a browser and seeing first
audio at 1796ms against a 600ms deadline, then reasoning that a fixed phrase never needs a
network. The blueprint states it as settled practice. Arriving somewhere independently is
weaker evidence than measuring, and stronger than reading.

## Adopt, in priority order

Each of these is a defect we have now, not an enhancement.

### 1. Truncate the reply at the last word actually played

Blueprint 3.1 calls this "the one everyone botches": on barge-in you must truncate the
assistant turn in conversation state at the last word the user *heard*, not at the synthesis
buffer, or the model believes it said things nobody heard and coherence collapses two turns
later.

We drop stale chunks by generation, which protects the ear. We do nothing about the
transcript, which protects the model. It is latent only because each turn currently starts
fresh with no history; the moment conversation memory exists it becomes a correctness bug.
Needs word marks from the synthesiser and a played-position signal from the client.

### 2. Tell a backchannel from an interruption

Blueprint 3.1 signal 2, and the sharpest India-specific point in the document: Hindi and
Hinglish speakers backchannel constantly, "haan", "achha", "hmm", "theek hai", and killing
playback on every one "makes the agent feel neurotic".

Our barge-in fires on any sustained speech clearing `min_speech_ms`, so it interrupts on
"haan" today. Fix is a lexicon plus a duration gate: short and lexicon-matched keeps
talking, longer or content-bearing interrupts. We already run STT continuously, so the
lexical gate costs nothing new. This is the highest-value behavioural fix on the list.

### 3. Deterministic inverse text normalisation before the model sees text

Blueprint 3.3 item 3: "never delegate numeric normalisation to the LLM". Their example is
"sava crore" to Rs 1.25Cr. Ours is worse, because the number *is* the answer: an income of
"pachaas hazaar" mis-normalised to 50 or 5,00,000 changes an eligibility verdict.

A rule pass between transcript and prompt, in code, tested. We already refuse to let the
model invent thresholds; this stops it mis-reading the applicant's own figure.

### 4. Contextual biasing on the recogniser

Blueprint 3.3 item 2, costed at 0ms: load domain vocabulary into the recogniser's biasing
channel and "this single lever often moves entity accuracy more than switching vendors".
Groq's transcription endpoint takes a `prompt` parameter that does exactly this. Scheme
names are our entity class: PM-KISAN, Ayushman Bharat, Ujjwala. Free accuracy, no latency.

### 5. A validator between the model and the synthesiser

Blueprint 5.2, and closing truth 3: "probabilism stops at the guardrail line". The model
emits intents; deterministic validators check them against the source of truth *before*
synthesis. Their exposure is RERA claims; ours is telling somebody they qualify for a
payment they do not.

`vaani/tools.py` refuses to invent a threshold, but nothing stops the model stating a number
in prose that the tool never returned. A check that every figure in the reply appears in the
tool result, run before TTS, closes that. This one protects a person rather than a metric.

### 6. Confirmation as a dialog act on high-stakes slots

Blueprint 3.3 item 4: treat a recognised entity as a hypothesis, and below a confidence
threshold confirm it verbally. "One confirmation turn is cheaper than one wrong site visit."
For us, one confirmation turn is cheaper than one wrong eligibility answer, and it costs a
turn of latency we would otherwise be optimising for nothing.

### 7. Acknowledge after an interruption rather than going silent

Blueprint 3.1 recovery UX: do not restart the abandoned sentence, re-plan and acknowledge
briefly, "haan, boliye". We currently return to listening with no audio at all, which reads
as the agent having given up.

### 8. Record where users interrupt

Blueprint 3.1: an `interrupted_at` marker is "your best script-optimisation signal". We
record that a turn was interrupted, not where. Cheap to add, and it is analysis the ablation
can use.

### 9. Prompt caching, and prompt order

Blueprint 3.2 layer 5: pin the static prefix and put volatile state after it, for 30 to 60%
off time to first token. Our system prompt is static and sits first already, so this is a
provider-support question rather than a redesign, and it is measurable either way.

### 10. A real VAD instead of frame energy

TEN ships TEN VAD; LiveKit and Pipecat both use Silero. Energy against a threshold cannot
tell speech from a fan, which we already documented as a known limit. Silero is ONNX on CPU
at about 1ms a frame. This is the M1.1 upgrade, and it carries the same model-download
question as Piper, so it waits on that decision rather than on merit.

## Take the framing, not the mechanism

Three ideas that change how we report rather than what we build.

**Tail multiplication.** Blueprint 1.1: with six serial stages each at p95 only 5% of the
time, about 26% of turns contain at least one p95 stage, and "users judge you by your worst
turns". This is arithmetic we can verify against our own traces, and it is the clearest
argument for why a p95 floor is a different problem from a p50 target. It belongs in the
ablation write-up.

**Endpointing is a Pareto frontier, not a bug.** Blueprint 1.1 item 2. Independent support
for reporting false-endpoint rate beside the milliseconds rather than instead of them.

**Latency is an architecture property, not a model property.** Closing truth 1. Consistent
with what we measured: the two largest wins so far were removing a network call from the
filler and correcting where a clock started, neither of which was a model change.

## Reject, with reasons

Not oversights. Each contradicts a stated non-goal, and saying so keeps the scope honest.

- **Telephony, SIP, DLT, carrier codecs, jitter buffers, server-side AEC.** SPEC non-goal 1
  is browser microphone only. The blueprint's 8kHz narrowband analysis and its word error
  rates of 18 to 35% do not apply to a WebRTC capture at 16kHz, and quoting them would be
  borrowing someone else's difficulty.
- **Speech-to-speech.** Non-goal 2, and the reason is the project's whole contribution: a
  single model has no stages to measure.
- **Self-hosted GPU inference, regional model hosting.** Non-goal 7. Worth noting the
  backend already runs in Singapore rather than the US, which is the cheap end of the
  blueprint's geography argument.
- **Kafka, ClickHouse, Redis session store, worker autoscaling, warm pools.** One session at
  a time is the design point (A10), and there is no database at all by choice.
- **Multi-party, multi-session negotiation state.** Non-goal 8.
- **Visual workflow builders** (Dograh, TEN's TMAN Designer). A builder is a product surface
  for people who are not writing the pipeline, and the pipeline is the artifact here.

## Where we can actually be better

Not on infrastructure. On evidence.

1. **A clock definition that costs us rather than flatters us**, measured from the last frame
   of user speech, hashed in `bench/stages.md` before anything was measured. We got this
   wrong ourselves once and fixed it; it is now pinned by a test.
2. **Filler audio counted separately from answer audio.** The blueprint separates them in its
   tool-turn row, which is more than most; we enforce it in code so a configuration cannot
   meet a target by learning to say "achha" quickly.
3. **Per-technique deltas with intervals, including the ones that bought nothing.** No
   project in this list publishes that. It is the entire Depth Chapter.
4. **A published corpus and a script a stranger can re-run.** Every number regenerable, raw
   per-run data included, so somebody can compute a different statistic or catch an error in
   ours.

## Open questions this reading raised

- Does Groq's transcription endpoint accept a biasing prompt, and does it measurably move
  scheme-name accuracy? Testable now, and it is free.
- Does our provider support prompt caching, and what does it do to time to first token?
- What fraction of real Hinglish barge-ins are backchannels rather than interruptions? The
  answer sets the lexicon and the duration gate, and it needs the recorded corpus.
