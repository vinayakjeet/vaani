# M5.5: threats to validity, in the order that matters most

Written after M5.2 and M5.2c, not before, because most of these were not
hypothetical when this list was started: they are the specific things that
went wrong while measuring, named here rather than folded quietly into a
caveat at the bottom of a table.

## 1. This session's own testing volume degraded the thing it was measuring

The most concrete threat here is not a textbook one. Three consecutive n=20
ablation attempts on 2026-08-19, the third paced at 4 seconds between every
turn specifically to rule out a bursty per-minute token window, all failed
the same way: every streamed-arm turn timed out and the unstreamed arm
"succeeded" at ten times its ordinary latency. Root-caused via the response's
own rate-limit headers to account-level throttling from several hundred real
calls this session had already made that day, not a code defect. Full account
in QUOTAS.md and DECISIONS.md.

The threat this leaves standing: any number measured late in a long testing
session, on this project or the next one, may be measuring the account's
accumulated load that day as much as the code under test. A single-machine,
single-account project has no way to separate "the pipeline is slow" from
"this API key has made four hundred calls since breakfast" without a second,
untouched account to compare against, which this project does not have. The
numbers published in README and BACKLOG were measured before this session's
own volume caught up with it; a reader re-running `bench/ablation.py` late in
their own heavy testing day should expect worse numbers than the ones
published here, for reasons that have nothing to do with the code.

## 2. The published streamed-vs-unstreamed number predates its own fix

M5.2c's filler-interruption fix is implemented, unit-tested, and deployed.
The n=20 re-verification its own acceptance line asks for is blocked by
threat 1 above, not done. The ablation number currently published in README
and BACKLOG's M5.2 entry is the *pre-fix* measurement: streaming lost to the
naive baseline because the filler played to its own exhaustion regardless of
when the real answer was ready. Whether that fix actually closes the gap, or
narrows it, or leaves some other term dominant, is not yet known from a live
number. Treat the published ablation table as characterising the code as it
stood before M5.2c, not the code as currently deployed.

## 3. One provider set, one of them free and unofficial

The whole ablation runs against Groq's free tier and `edge-tts`, an
unofficial API with no published SLA and no quota. Every millisecond in
these tables is specific to this pair. A paid Groq tier, a different STT/LLM
provider, or a licensed TTS vendor would plausibly change not just the
absolute numbers but which stage dominates the waterfall: `bench/waterfall.py`
consistently found `stt.stream` and `llm.generate` as the two widest terms
on this stack, and that ordering is not guaranteed to hold on a different
one. M4.4 built and live-verified the second stack (Sarvam Saaras and
Bulbul) but the full paid side-by-side comparison was never run, deliberately,
pending a balance check this session could not do for the user; see BACKLOG.

## 4. One machine, one network, one country

Every measurement here was taken from one Windows machine on one network
path in one geography, against providers whose own infrastructure has its
own regional latency profile. Render's deployed instance runs in Singapore
(see QUOTAS.md); a caller elsewhere pays a different transport cost than
this project's own bench scripts do, which call the providers directly and
never touch the deployed service's own network hop at all. Nothing here
measures what a real caller on a mobile network in a different region would
experience end to end.

## 5. Recorded corpus, not live human speech

`bench/corpus/manifest.json` is synthesised via TTS, not recorded from a real
speaker. It carries none of what real speech has and a fixed script cannot:
disfluency, self-correction, background noise, a genuine mid-sentence pause
a listener would call thinking rather than finishing. `bench/endpointing_frontier.py`
states this limitation explicitly rather than filling the gap with a guess:
its "cut/tested" table is a proxy for how each VAD aggressiveness setting
would behave against a battery of synthetic pause durations, not a measured
false-endpoint rate, because no corpus of real disfluent speech exists here
to measure one against.

## 6. The model's own non-determinism, at every n this project used

Three independent live runs of the same 50-scenario eval set, same labels,
same code, scored 40/50 each time but not the *same* 40: the specific
scenarios that failed shifted a little run to run (see BACKLOG's M4.7 entry
for the category breakdown of each). The ablation's own n=20 is the
pre-registered size specifically because n=2 and n=3 runs earlier in this
project disagreed with each other in *direction*, not just magnitude, before
the ablation harness was even fully debugged (see DECISIONS.md, the ablation
clock bug entry). Twenty per arm is enough to see a stable direction on this
stack; it is not enough to treat any single run's exact median as more
precise than roughly ±10%, and no formal confidence interval is published
here, on purpose, rather than computed from too few points to mean much
beyond the p95 already reported.

## 7. The clock definitions and the code that measures against them share one author

`bench/stages.md` defines what every stage span starts and ends at, and the
same session that wrote that definition also wrote the code that measures
against it. There is no independent second implementation cross-checking
`TurnClock`, `speak_within`, or the endpointer's own backdating logic. Three
real clock and measurement bugs were found and fixed this project (the OTel
context leak, the unstreamed baseline's skipped endpoint wait, the ablation's
own crash-on-one-bad-turn harness gap), each caught by noticing a result
contradict either a pre-registered prediction or an internal consistency
check, not by an outside reviewer. A bug that produces a plausible-looking
number, one that does not contradict anything, would not have been caught by
this project's own process, and this project has no reason to believe it has
found every one that exists.

## What this list is not

Not a reason to distrust every number here equally; the waterfall and
tail-multiplication figures do not depend on account throttling the way a
60-second-timeout-bounded ablation run does, and the eval's 80% held stable
across three genuinely independent live attempts, all before this same
session's own load caught up with it and produced threat 1, which is itself
evidence against threat 1 explaining that particular number. Each figure in
README carries its own caveat where one applies; this file is the place
those caveats point back to, not a blanket discount on all of them.
