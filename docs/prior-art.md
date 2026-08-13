# Prior art, and what Vaani takes from it

Read against the TrueState voice-agent blueprint (July 2026) in full, the READMEs of TEN
Framework and Dograh, and what the blueprint reports about LiveKit Agents, Pipecat, Bolna
and Vocode.

**What this document is not based on.** No research papers were read. No source was read for
Bolna, Vocode, LiveKit or Pipecat: their mechanisms here are secondhand, from the blueprint's
comparison tables. An earlier version of this line claimed "the turn-taking literature", which
was not earned and is corrected here. Where a claim below rests on secondhand reporting rather
than something read directly, treat it as a lead to verify rather than a finding.

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

### 9. Prompt caching, and prompt order (M1.14)

Blueprint 3.2 layer 5: pin the static prefix and put volatile state after it, for 30 to 60%
off time to first token. Our system prompt is static and sits first already, so this is a
provider-support question rather than a redesign, and it is measurable either way.

Answered in the third pass: Groq does it automatically. Tracked as **M1.14**. An earlier version
of this section called it M1.9, which is a different and already-finished item, and no task
existed at all until a backlog audit found the gap.

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


## Second pass, 2026-08-13: primary sources

The first pass rested on one document. This pass read Bolna's source directly and searched
the 2026 literature. Two of my earlier recommendations were wrong and are corrected below.

### Read directly

`bolna/agent_manager/interruption_manager.py` (512 lines), `bolna/helpers/mark_event_meta_data.py`,
`bolna/constants.py`. Pipecat and LiveKit source still not read; their READMEs do not describe
internals.

### Correction 1: do not enumerate backchannels, enumerate stop words

I told you to build a backchannel lexicon so "haan" does not interrupt. Bolna does the
opposite and it is better.

`should_trigger_interruption` interrupts when `word_count > number_of_words_for_interruption`
(default **3**) **or** the transcript matches `ACCIDENTAL_INTERRUPTION_PHRASES`, which is not
a backchannel list at all: it is "stop", "wait", "no", "hold on", "enough", "excuse me",
"not now", "stop speaking". Short *stop* words that must interrupt despite being short.
And `is_false_interruption` discards a final transcript that is short and not on that list.

The insight is that backchannels are an open set and stop words are a closed one. Enumerating
"haan, achha, hmm, theek hai, accha ji, ji, hm" is a losing game in Hinglish; enumerating the
handful of short phrases that mean stop is tractable. Word count carries the rest.

### Correction 2: a three-state audio gate, not two

We have SEND or BLOCK by generation. Bolna has **SEND, BLOCK, WAIT**:

- BLOCK when the sequence id is not in the valid set, which is our generation check.
- **WAIT while the user is speaking**, holding audio rather than discarding it.
- **WAIT during a grace period** after the user's utterance ends, `incremental_delay` default
  **900ms**, and only after the first two turns so the greeting is not delayed.

WAIT is the state we lack, and it is what stops the agent talking over someone who has
started again. Also worth stealing: `sequence_ids` is a **set** with -1 reserved for
background audio, so several streams can be valid at once. Our single integer cannot express
that.

### The playout problem, solved concretely

`mark_event_meta_data.py` answers the question our `vaani.playback.queued_ms` currently
fudges. Their comment is the lesson: audio is handed over faster than real time, so a chunk
starts playing when the previously queued audio ends, not when it was sent. Hence

    audio_playing_until = max(audio_playing_until, now) + duration

That gives a playout estimate with no client acknowledgement at all, which we could adopt
today. On top of it they track `record_heard_text` per turn, accumulating the text of chunks
the far end acknowledged, so `get_heard_text_for_turn()` is literally what the user heard.
That is the truncation input the blueprint said everybody botches, and it is about forty lines.

### Metrics worth copying outright

- `tts_speed_ratio = total_audio_duration / wall_clock`. Below 1 means synthesis cannot keep
  up with playback and the answer will stutter. One division, and it catches a failure our
  latency numbers cannot see.
- `high_delay_count`, chunks whose acknowledgement exceeded `HIGH_DELAY_THRESHOLD = 2.0`.
- `total_missed`, marks sent but never acknowledged.
- **`longest_agent_monologue_ms`** and a talk-to-listen ratio. Not latency at all, and closer
  to whether a conversation felt human than anything in our current budget.
- `cleared_on_interrupt` per chunk, so an interrupted turn can be reconstructed afterwards.

### Speculative prefill, as actually built

Bolna wires Deepgram Flux with three constants: `EOT_THRESHOLD = 0.7` to declare end of turn,
`EAGER_EOT_THRESHOLD = 0.5` to **fire the LLM early**, and `EOT_TIMEOUT_MS = 500` as the
forcing timeout. There is an `eager_llm_task` that is cancelled when the guess turns out
wrong. That is SPEC's speculative prefill technique with a working shape: two thresholds on
one confidence signal, and a cancellable task.

Our recogniser emits no such confidence, which is the real blocker for that technique, not the
prefill logic.

### 2026 literature

Searched and read. All are 2026 unless noted.

**X2-Turn** (arXiv 2608.10878), the closest thing to what we should want. One model, two heads
on shared hidden states: streaming ASR tokens and turn state in a single forward pass at
**80ms frames**. Five states: idle, noidle, incomplete, complete, **backchannel**. At
tau = 480ms it reports **91.0% complete accuracy at 288ms latency** for Chinese and 92.1% at
225ms for English, beating SoulX-Duplug's 77.67%. Turn predictions deliberately do not feed
back into ASR decoding, so a turn error cannot corrupt the transcript. Reimplementing it needs
26k hours of ASR plus forced alignment and LLM-labelled word-level states, so it is a
reference architecture rather than something we build.

**smart-turn** (pipecat-ai, BSD-2). This one we can actually use: Whisper-tiny backbone,
**8M parameters, 8MB quantised int8, 10ms to 100ms on CPU**, input is **16kHz mono PCM up to
8 seconds**, which is exactly the format our worklet already produces. It changes the
model-download question completely: 8MB is not the 60MB Piper problem.

**RelayS2S** (arXiv 2603.23346): dual-path speculative generation reports **P90 onset latency
of 81ms against 1091ms for a cascaded baseline**, keeping 99% of the cascade's quality score.
Worth knowing as the honest ceiling on what a cascade can reach.

**SoulX-Duplug** (arXiv 2603.14877): a plug-and-play streaming state predictor folding VAD,
ASR and turn detection into one module, because non-streaming turn detection adds latency that
grows with input length.

**Speculative End-Turn Detector** (arXiv 2503.23439, 2025) and **Discourse-Aware Dual-Track
Streaming Response** (arXiv 2602.23266) round out the direction: turn state is being predicted
jointly with recognition rather than bolted on after silence.

**Gradium's product note** describes emitting turn-completion predictions every 80ms as
inactivity probabilities at 0.5, 1, 2 and 3 second horizons. A forecast rather than a verdict,
which is a more useful shape than our boolean.

### What this changes about our plan

The direction of the whole field is that **VAD plus a silence timer is the thing being
replaced**, and our `completeness.py` word-order rule is a hand-written approximation of a
model that now exists at 8MB. The rule is still worth keeping as the ablation's baseline arm,
which is exactly what SPEC wants, but it should not be the only arm.


## Third pass, 2026-08-13: a wider web sweep

Searched rather than assumed. Fourteen papers and thirteen tools below, with the four findings
that change the plan first.

### Finding 1: Piper is the wrong local synthesiser, and M1.8 was built on a false premise

M1.8 exists to put a local CPU synthesiser on the critical path so the TTS tail stops being a
network tail. Measured benchmarks say Piper is **1510ms to first audio with a 2.6GB peak**,
which is worse than the hosted voice it was meant to replace.

The right model is **Kokoro**: 82M parameters, **97ms time to first byte** on CPU baseline, RTF
0.03 on GPU, and it is consistently the lowest first-audio latency across GPU tiers in
Picovoice's and CodeSOTA's comparisons. **Orpheus** sits at 187ms with higher fidelity.

So the technique is right and the model was wrong. Had we built M1.8 as written it would have
made p95 worse and we would have concluded local synthesis does not help.

### Finding 2: SPEC A7 is answerable, with a real measured number

A7 says the 1.4 to 1.7 second industry median is uncited and must be sourced or dropped.
Openbenchmarks publishes **time to first audio byte measured from the call's own audio across
five commercial platforms over 2078 usable turns**, lowest median **1296ms (Telnyx)**, with
ElevenLabs, Bland, Vapi and Retell alongside. That is a dated, first-party-measured,
reproducible figure rather than a vendor claim, and it is the comparison the chart should use.

It also reframes our target honestly: p50 under 500ms would be roughly **2.6 times faster than
the best measured commercial median**, which is a strong claim and should make us more
suspicious of our own numbers, not less. Their measurement includes PSTN transport we do not
pay, so the comparison must state both clocks.

### Finding 3: Groq prompt caching is automatic, so M1.9 is already half done

Prompt caching on GroqCloud applies to all requests to supported models with **no code changes
and no extra fee**, cuts latency and input cost by about **50% on the cached prefix**, and
expires after two hours of disuse. Our system prompt is static and already sits first, which is
the ordering that makes a prefix cacheable, so the remaining work is measurement rather than
design.

### Finding 4: barge-in should be stopped in the browser, not on the server

X-Talk describes a **VAD-driven preemption loop that pauses client playback the moment speech is
detected**, before any server decision. Our barge-in waits for a round trip: detect on the
server, cancel, stop sending. Pausing locally on the client's own level detector, then letting
the server confirm or resume, puts a 100ms budget within reach for free and is a dozen lines in
`web/index.html`.

### Papers

1. **X2-Turn**, arXiv 2608.10878. Dual-head streaming ASR plus turn state, 80ms frames, five
   states including backchannel and incomplete, 91.0% complete accuracy at 288ms.
2. **RelayS2S**, arXiv 2603.23346. Dual-path speculative generation, P90 onset 81ms against
   1091ms cascaded, 99% of cascade quality retained.
3. **SoulX-Duplug**, arXiv 2603.14877. Plug-in streaming state predictor folding VAD, ASR and
   turn detection together, because non-streaming turn detection adds latency that grows with
   input length.
4. **Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models**, arXiv 2606.11167.
   Names the four axes of interactivity: pause handling, turn-taking, backchanneling, user
   interruption. A better checklist for conversational quality than our latency budget.
5. **Aligning Backchannel and Dialogue Context Representations**, arXiv 2604.16622. Contrastive
   fine-tuning for backchannel appropriateness.
6. **RESPOND**, arXiv 2603.21682. Frame-wise prediction of **when and what** backchannel to
   produce, treating backchannels as opportunities rather than reactions. Its user studies,
   notably with older adults, report higher perceived naturalness when the agent both allows
   barge-in and offers timely acknowledgements.
7. **The Silent Thought**, arXiv 2603.17837. Latent reasoning inside a full-duplex model so it
   can backchannel and yield gracefully.
8. **Discourse-Aware Dual-Track Streaming Response**, arXiv 2602.23266.
9. **Enabling Conversational Behavior Reasoning in Full-Duplex Speech**, arXiv 2512.21706.
10. **X-Talk**, arXiv 2512.18706. Argues modular speech-to-speech is underrated, and contributes
    the client-side preemption loop in finding 4.
11. **Speculative End-Turn Detector**, arXiv 2503.23439.
12. **On the Landscape of Spoken Language Models**, arXiv 2504.08528. Survey. States that
    dialogue quality is determined by interactivity measures rather than by transcription
    accuracy, which is an argument for M4.10.
13. **VoiceAgentEval**, arXiv 2510.21244. Dual-dimensional benchmark for voice-agent evaluation.
14. **HiACC**, ScienceDirect S2352340925006109. The first code-switched Hinglish corpus with both
    adults and children: 3318 adult and 1858 child segments, 5.24 hours, read and spontaneous.
    Directly relevant to M4.6, and a published comparison point for our own eval set.

Also recorded: code-switched speech raises word error rate by a **relative 30 to 50%** against
monolingual input, which quantifies SPEC A4's hypothesis rather than leaving it as an assertion.

### Tools

1. **smart-turn** (pipecat-ai, BSD-2). 8M params, 8MB int8, 10 to 100ms CPU, 16kHz mono PCM.
2. **Silero VAD**. The default in both LiveKit and Pipecat.
3. **TEN VAD** and **TEN Turn Detection**, shipped separately from the TEN framework.
4. **Kokoro** TTS. 82M params, 97ms TTFB, RTF 0.03. The local synthesiser M1.8 should use.
5. **Orpheus** TTS. 187ms TTFB, higher fidelity, also served serverless by Together.
6. **Piper**. Recorded as the negative result in finding 1.
7. **IndicWhisper**, Apache 2.0. Whisper fine-tuned on Indic speech.
8. **Sarvam Saaras V3**. 19.31% WER on the ten most popular IndicVoices languages.
9. **Sarvam-1**, open weights base model.
10. **AI4Bharat, Bhasini, IndicTTS**. The Indic open-source stack around Sarvam.
11. **Pipecat**, **LiveKit Agents**, **Bolna**, **Vocode**, **Dograh**. Orchestration, covered above.
12. **aiewf-eval**. Open-source voice-agent evaluation covering latency, tool calling,
    instruction following and knowledge grounding across long multi-turn conversations.
13. **Openbenchmarks TTFAB**. The external measured baseline in finding 2.

### The framing worth stealing

Two ideas reorder our priorities rather than adding to them.

**Interactivity, not latency, is the quality measure.** The survey and the interactivity-alignment
paper both define spoken dialogue quality as pause handling, turn-taking, backchanneling and
graceful interruption. Our entire budget measures one of those four. That is not wrong, it is
narrow, and M4.10 is the beginning of a fix.

**Backchannels are something to produce, not only to survive.** We have been treating "haan" as
a thing that must not interrupt us. RESPOND treats the agent emitting a timely acknowledgement as
a measurable gain in perceived naturalness. That inverts the item and is cheap for us: the filler
bank already exists.


## Fourth pass: `sync_history`, read line by line

125 lines of `task_manager.py` read directly. This is how Bolna reconciles the conversation
record after an interruption, and it is the most carefully-guarded code in the repo. Five
things in it that we would each have got wrong.

### 1. Evidence versus blind fallback, and refusing to trim on the latter

They resolve which turn to trim through a chain, then keep a separate flag,
`target_from_evidence`, recording whether the answer came from **actual evidence** (pending
marks, or acknowledged text) or from a **blind fallback** to "the latest assistant turn".

If it came from the fallback, they **refuse to trim at all**. Their comment says why: a second
cleanup after a previous one already committed a filler would otherwise delete that filler from
history. So the guard is not about correctness of the trim, it is about idempotence of repeated
cleanups.

We have no equivalent because we have no conversation record yet. The moment we add one, this
is the bug we would ship: a second interruption with nothing pending silently deletes the last
committed message.

### 2. A four-level fallback for "what was heard", ordered by trust

In order: acknowledged text for this response id, then for this turn id, then the mark store's
text for the response, then for the turn, and only then a global accumulator.

The global is used **only when the target turn is unknown**, and the comment explains the
trap: with a known target turn, the global can hold stale text from a previous turn, for
instance acked post-tool audio, and using it would corrupt the wrong message. A fallback chain
that gets less specific as it descends must stop before it becomes wrong rather than merely
vague.

### 3. Duration-proportional reconstruction when nothing was acknowledged

If no acknowledged text exists at all, they estimate what was played from timing:

- `actual_play_time = interruption_timestamp - first_chunk_sent_ts`
- walk the pending chunks accumulating their durations
- chunks entirely inside that window count as played in full
- for the chunk **straddling** the boundary, take `proportion = remaining_time / chunk_duration`,
  slice that fraction of the characters, and then **trim to complete words**

Characters proportional to time is a crude model of speech and it is obviously better than the
two alternatives, which are dropping the straddling chunk entirely or keeping all of it. This
is the mechanism we would need if we never add client acknowledgements at all, and it makes
M2.7 implementable without any client change.

### 4. Streaming synthesisers do not populate the text, so there is a second path

Marks from a streaming synthesiser carry audio duration but no `text_synthesized`, so the whole
text-based reconstruction is unavailable. They group pending marks by turn id and work from
duration alone. Two code paths for the same question because the providers differ, which is
exactly the seam our `SttProvider`/`TtsProvider` interfaces are meant to hide and cannot here.

### 5. Backchannel audio is excluded from the played-text reconstruction

`mark_type in ["pre_mark_message", "backchanneling"]` is skipped. An acknowledgement the agent
emitted is audio the user heard, but it is **not part of the reply**, so counting it would put
"haan" into the assistant's turn and shift every subsequent character offset.

This lands directly on M2.12: the moment we emit backchannels, they must be marked as a
different kind of audio or they will corrupt the transcript we are trying to protect.


## Fifth pass, 2026-08-13: the orchestration end to end, and papers as full text

The earlier passes read three files of Bolna and a set of abstracts. This one read the
orchestration properly and pulled full texts where they exist. Two of the previous pass's
claims were overstated and are corrected first, because a note that flatters a source is
worse than no note.

### Read directly this time

`bolna/agent_manager/task_manager.py` in the regions that decide behaviour:
`__process_output_loop`, `__listen_synthesizer`, `_handle_transcriber_output`, `sync_history`,
`__cleanup_downstream_tasks`, `_stage_assistant_history` and its pair,
`update_transcript_for_interruption`, `__check_for_backchanneling`,
`compute_last_ai_audio_timestamp`. Plus `interruption_manager.py`,
`mark_event_meta_data.py`, `output_handlers/default.py`, `constants.py`, and the Flux
event handling in `transcriber/deepgram_transcriber.py`. Pipecat and LiveKit source still
not read.

### Correction 3: RESPOND's user study is six people, and its own participants disliked the feature

The third pass wrote that RESPOND's user studies "notably with older adults, report higher
perceived naturalness". The full text says the pilot is "six participants (4 female, 2
male)" from the lab community, and there are no older adults in it. Worse for the claim we
built on it: participants reported that "excessive backchannels, especially mid-sentence
ones, were considered distracting and unnatural", with only sentence-final ones felt as
natural. Backchannel is also the weakest of its three classes at 0.697 F1 on MM-F2F against
0.770 and 0.773 for the others.

The item survives as M2.16 with a constraint taken from the finding instead of the framing:
emit only at a boundary, and report the rate. This is the second time in this project that
reading past an abstract weakened a recommendation, which is an argument for the habit
rather than a complaint about it.

### Correction 4: our own playout estimate was wrong, and Bolna's code says why

`output_handlers/default.py` has `_playout_duration`, which returns **zero** rather than a
number when the format is not in `UNCOMPRESSED_AUDIO_FORMATS`. Its comment: compressed audio
"would otherwise report a confidently wrong duration, which the completion watchdog's playout
estimate would then trust".

We adopted their playout line in M2.10 and then fed it MP3 byte counts divided by 30000, a
constant with no derivation anywhere. edge-tts is 48 kbit/s constant bitrate, which is 6000
bytes a second, so the estimate ran five times fast and the 900ms grace period expired after
180ms. Recorded in LEARNING.md. Adopting a mechanism from a codebase and not the guard beside
it is a specific and repeatable failure mode.

### The mechanism we missed twice: stage the turn, commit it on send

The most useful thing in the whole file, and it is not in `sync_history` at all.
`_stage_assistant_history` holds the assistant's text keyed by sequence id;
`_commit_staged_assistant_history` writes it to the conversation record at the moment its
audio clears the SEND gate; the BLOCK path calls `_drop_staged_assistant_history` instead.

So the ordinary interrupted turn needs no reconstruction: text whose audio never reached the
transport never enters the record. The duration-proportional reconstruction the fourth pass
admired is the *fallback* for a turn that was half spoken. We had read the hard path and
missed the easy one, and the easy one is what makes the hard one rare. Tracked as M2.7b, and
it is the reason to decide the conversation record's shape now rather than after building it.

### Guards worth having read

- **A stuck flag is repaired, not stepped over.** The WAIT branch of the output loop checks
  `user_speech_staleness_s()` against `STUCK_AUDIO_GATE_RELEASE_S = 3.0` and calls
  `on_user_speech_ended(update_utterance_time=False)`. Our version forced the chunk through
  and left the flag set, so every later chunk waited again. Same timeout, arrived at
  independently; different thing done with it. In LEARNING.md.
- **Control signals must still flow on the blocked path.** The end-of-stream byte is sent
  through `handle()` even when the audio is discarded, because otherwise the final mark is
  never created, no echo arrives, and `is_audio_being_played` latches true for the rest of the
  call. A whole class of bug: the drop path skips the message that clears the state.
- **`update_transcript_for_interruption` returns empty when the heard text is not found in
  what was spoken.** The sixth guard, missed by the fourth pass. Stale turn data corrupts the
  record quietly; losing it is visible.
- **The global "what was heard" accumulator is used only when the target turn is unknown**,
  and `target_from_evidence` refuses to trim at all when the target came from a blind
  fallback. Both already recorded; both confirmed by reading the code rather than the summary.
- **`compute_last_ai_audio_timestamp` clamps the playout estimate to now**, so measured
  silence can only shrink. Their comment states the safety property explicitly: it can delay
  a hangup but never cause an earlier one. A good habit for any estimate feeding a watchdog.

### The speculative-prefill blocker, and what actually unblocks it

The fourth pass concluded that our recogniser emits no confidence, so SPEC's speculative
prefill has no signal to fire on. The Flux handling confirms the shape: `EagerEndOfTurn`
above 0.5 starts a cancellable `eager_llm_task`, `TurnResumed` cancels it, `EndOfTurn` above
0.7 is the real thing, `EOT_TIMEOUT_MS = 500` forces the decision, and an empty `EndOfTurn`
with an eager guess outstanding cancels it. There is also a stall floor so a turn cannot stay
open forever: `max(3.0, eot_timeout * 4)`.

What the fourth pass did not notice is that **smart-turn supplies exactly that signal**. It
returns a probability, not a boolean. Two thresholds on it plus a cancellable task is Bolna's
mechanism with our own model, and it needs no vendor with a streaming socket. That makes
M1.12 a prerequisite for the speculative-prefill row rather than a parallel improvement.

### Papers, as full text

**X2-Turn** (arXiv 2608.10878), read in full. Five frame-level turn states, and the naming
matters: `idle`, `noidle` for "active speech without semantic content (e.g. the initial
syllables of an utterance)", `incomplete`, `complete`, `backchannel`. 80ms frames on a
Voxtral Realtime backbone. Latency is defined as `L = tau - (e_i - s_i)`, so it can be
negative. At tau = 480ms: Chinese 91.0% complete, 93.0% incomplete, 96.0% backchannel, 288ms
latency; English 92.1% complete, 84.6% incomplete, 225ms. At tau = 320ms: Chinese average
90.67% at **120ms**, English 85.09% at **65ms**. Training is 26k hours of ASR (14k Chinese,
12k English) plus 126 hours of EasyTurn and 249 hours of Fisher, labelled with
Qwen3-ForceAligner for word timestamps and Qwen3.5-Plus for word-level state.

Two things to take. Their tau table is the reporting shape our VAD knob should have, one knob
with both axes and no argument about the right setting: tracked as M4.13. And turn
predictions deliberately do not feed back into ASR decoding, so "turn state prediction errors
do not affect subsequent ASR decoding". Our partials feed the endpointer and never the
reverse, so we already hold that property; worth knowing it is a design choice somebody
defends rather than an accident.

**X-Talk** (arXiv 2512.18706), read in full, and the most directly useful paper in the
sweep. Client-side Silero pauses TTS playback the moment speech is detected, then lightweight
semantic verification either terminates synthesis and inference or resumes the suspended
playback, with the frontend notified to resume on a false interruption. Their
false-interruption filter is three rules: minimum audio length 0.5s, empty ASR output or a
single character or digit, and a transcription containing only filler words. That is M2.6's
design and M2.11's, arrived at from a different direction.

Their latency table also gives us the honest comparison SPEC A7 wants: 284ms and 272ms at 5s
input, up to 420ms and 610ms at 30s, with the explicit statement that the frontend VAD's
"500 ms end-of-utterance silence threshold ... is not included in latency calculation". On
our clock that is 772ms to 1110ms. A dated, first-party, reproducible figure that needs no
vendor's marketing number. Tracked as M4.14.

**RESPOND** (arXiv 2603.21682), read in full. Three classes: turn claim, backchannel, stay
silent. 5-second windows, 50ms stride, word-level granularity, bounded in practice by 250 to
500ms of ASR latency. Two knobs in [0,1] with named presets: backchannel intensity 0.1
passive and 0.6 collaborative, turn-claim aggressiveness 0.0, 0.2, 0.8. Accuracy 0.756 on
MM-F2F and 0.87 on CANDOR. See correction 3 for what the user study actually says.

**RelayS2S** (arXiv 2603.23346). A fast duplex S2S model drafts a response prefix streamed
straight to TTS while a slower ASR-to-LLM path generates the continuation, with a learned
verifier controlling the handoff. Described by its author as a "lightweight, drop-in addition
to existing cascaded pipelines" requiring no architectural modification to either component.
P90 onset comparable to the S2S model at 99% of cascaded quality. Still a reference rather
than a build: the fast path is a duplex speech model, which is non-goal 2 and needs a GPU.

**Multi-Faceted Interactivity Alignment** (arXiv 2606.11167). Names the four canonical axes:
pause handling, turn-taking, backchanneling, user interruption. Evaluated offline and in
real-time multi-turn dialogue on Moshi and PersonaPlex, with an LLM-based reward for response
quality included to stop interactivity gains degrading answers. That last detail is the same
guard as our non-goal 15 and M5.2b: optimise interactivity, measure the answer, or you will
trade one for the other without noticing.

**Speculative End-Turn Detector** (arXiv 2503.23439, ACL 2026). A lightweight GRU locally
finds silences; a Wav2vec model server-side decides whether a silence is a turn end or a
pause. The two-tier split is the same shape as our duration threshold plus verification, at a
different point in the pipeline. Code and data "available after review", so nothing to adopt.

**VoiceAgentEval** (arXiv 2510.21244). Six business domains, 30 sub-scenarios, 12 models,
automated plus human-in-the-loop across task execution, professional knowledge, adaptability
and user experience. No metric definitions in the abstract or the landing page; would need
the PDF to be useful, and it is a benchmark we are not scored against.

### Metrics with a threshold somebody else defends

Worth separating from the rest, because this sweep produced exactly one.
`barge_in_recovery_rate` in `interruption_manager.py` carries the comment "Hamming benchmark:
>90% good, <80% critical". Every other conversational-quality number found across two
sessions of reading is a ratio with no published bar. Its mirror,
`agent_interrupted_user_count`, counts the agent starting to speak inside the grace period
and being cancelled, which our latency numbers currently score as a success. Both in M4.12.
