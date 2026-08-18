# M5.1: the expected ordering, written before anything is measured

Hashed at the commit that adds this file, per M5.1's own acceptance: the prediction
exists in the log before the first M5.2 number does, so nothing here can be revised
to fit a result seen first.

## What this can actually ablate, and why the shape changed

BACKLOG named seven rows: streaming partials, first-sentence flush, speculative
prefill, VAD aggressiveness, semantic endpointing, local first-sentence TTS, tool
prefetch. Checked against the code rather than assumed, 2026-08-18: three of those
were never built. `git grep` for speculative prefill and tool prefetch finds them
only in a docstring, as future work; Kokoro (M1.8, local TTS) has no implementation
anywhere and its own BACKLOG entry says so, "not started," for a stated and costed
reason rather than an oversight. There is nothing to ablate because there is nothing
built, and pretending otherwise here would be inventing a row to fill a table rather
than reporting what exists.

Of the remaining four, two are not independently toggleable in the current code
either. Streaming transcription-while-speaking and first-sentence flush both live
inside `StreamingPipeline` as one integrated implementation; there is no seam that
turns one on while leaving the other off; only `vaani/turn.py`'s unstreamed baseline
and the full streamed pipeline exist as buildable configurations. So they are
measured together, as one delta against the unstreamed baseline, stated as bundled
rather than presented as two independent rows they are not.

That leaves three real, independently switchable arms:

1. **Streamed pipeline vs the M0 unstreamed baseline** (bundles streaming partials
   and first-sentence flush, per the above).
2. **Semantic endpointing on vs off** (`Endpointer(semantic=True|False)`).
3. **VAD aggressiveness, four named settings** (`Endpointer.at(0..3)`), already
   reported as a frontier rather than one setting in M4.13.

## Predicted ordering, largest time bought to smallest

1. **Streamed pipeline vs unstreamed.** Predicted the largest single win by a wide
   margin. The unstreamed baseline waits for the whole reply before synthesis
   starts at all; M1.13's own measurement already found the naive baseline cost
   roughly 2000ms just from that wait. Everything else here is tuning around an
   architecture that already overlaps stages; this is the architecture itself.

2. **Semantic endpointing on vs off.** Predicted second. `vaani/endpoint.py`'s own
   design note calls this the largest single term in the *optimised* budget: full
   trailing silence is up to 1000ms (aggressiveness 0) against roughly 200ms when a
   partial already looks finished. Predicted to buy several hundred milliseconds at
   p50, at a real, non-zero false-endpoint cost this run does not have a clean way
   to measure (M4.13's own stated limitation: no recorded speech with genuine
   mid-utterance disfluency exists to test it against).

3. **VAD aggressiveness, level 0 to level 3.** Predicted smallest of the three,
   not because the knob is unimportant but because it is a narrower range: the
   four settings span 1000ms down to 500ms of trailing silence, half of semantic
   endpointing's own already-measured spread, and it is one knob among several
   feeding the same trailing-silence budget semantic endpointing already touches.

## What would make this wrong

Semantic endpointing outscoring the streamed-vs-unstreamed delta would mean the
unstreamed baseline's own 2000ms figure does not hold on this corpus or this
model, which is itself worth knowing given the model swap earlier this session:
every number in M4.3 is dated against `openai/gpt-oss-120b`, not the model the
2000ms figure was measured against. If the streamed-pipeline delta is smaller than
expected, that is where to look first.
