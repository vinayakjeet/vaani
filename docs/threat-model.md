# Threat model: a public microphone endpoint

Vaani accepts audio from anyone who opens the page. The input is a person's voice
asking whether they qualify for a welfare payment, which means the untrusted input
and the sensitive data are the same bytes. That is the whole shape of the problem:
there is no version of this where the thing being validated can be discarded.

Short by design. It covers what crosses the socket, what is kept, and what somebody
who reaches the socket can do. `tests/test_redaction.py` is the enforcement, because
a threat model nothing checks is a paragraph.

## What crosses the socket

Inbound, from a browser nobody controls:

| Field | Classification | Handling |
|---|---|---|
| Audio frames, PCM16 | Sensitive. A person's voice, and their question | Held in memory for the length of one utterance, never written to disk |
| Frame generation, 2 bytes | Untrusted integer | Compared, never used as an index or a key into anything |
| Control messages | Untrusted string | Matched against a closed set; anything else is ignored rather than dispatched |

Outbound, to the browser:

| Field | Classification | Handling |
|---|---|---|
| Transcript text | Sensitive, and deliberately returned | The speaker's own words, shown so they can see what was heard. Leaves the process only on this socket |
| Reply text | Sensitive | Contains the eligibility answer, so it names the person's circumstances back to them |
| Audio chunks | Sensitive | Synthesised reply. Not retained after the send |
| Error reason and detail | Public | Exception class names and closed-set reasons, never provider bodies |

## What is retained

Nothing, after a session ends.

- Audio lives in one `bytearray` for the length of an utterance and is cleared when
  the turn ends. There is no disk write anywhere in the pipeline.
- Transcripts and replies are held for the length of a turn, in memory, and go out
  on the socket.
- No database. No user accounts, no history, no personalisation (SPEC non-goal 8).
- Spans and logs carry counts and durations only. Never the text, which is what the
  canary test enforces.

The one deliberate exception is the eval corpus, which is recorded speech committed
to the repository. It is scripted, spoken by the author, and contains no real
person's circumstances. That is a different artifact with different rules, and the
dataset header says so.

## What somebody who reaches the socket can do

**Occupy the service.** One session at a time is the design point (SPEC A10), so a
single connection holds the only slot. This is a denial of service against a free
demo and it is accepted rather than solved: the alternative is queueing, which
serves everybody badly and hides the fact that the server is full. The refusal is
explicit and says why.

**Send malformed frames.** Rejected at `decode` with a reason, and the session
continues rather than dropping frames until the transcript comes back as noise.

**Send a huge utterance.** Capped at `MAX_UTTERANCE_MS`. Without the cap a stuck
client streams until the instance dies, and on a free tier that is the whole
service rather than one session.

**Claim any generation number.** It is only ever compared against the server's own
counter, so the worst case is that the attacker's own audio is discarded. It is
never a lookup.

**Spend our provider quota.** Every accepted utterance costs a transcription, a
model call and a synthesis. This is the real exposure of a public demo on free
tiers, and the single-session limit is what bounds it.

**Make the model say something.** Prompt injection through speech is possible and
partly mitigated by the system prompt refusing to invent schemes, limits or
deadlines. It is not solved. The mitigation that matters is downstream: eligibility
answers come from `vaani/tools.py`, which validates arguments against a schema and
refuses an unknown scheme rather than improvising, so a model talked into a strange
answer still cannot invent a threshold.

## What is not a secret, and what is

The client bundle is two static files served from GitHub Pages. It carries the
backend URL and nothing else. Every provider key lives in Render's environment and
is read server-side; no key is ever sent to the browser, and there is no code path
that could.

`tests/app/test_tracing.py` and `tests/test_redaction.py` between them cover the
accidental-disclosure paths: the endpoint and auth header handling that once sent a
literal `Basic%20...`, and the exception messages that once carried provider bodies
into span events.

## Known gaps

Stated rather than implied, because a threat model that lists only solved problems
is marketing.

- **No authentication.** Anyone can use the demo, which is the point, and it is why
  quota exposure above is real rather than theoretical.
- **No rate limit beyond one session.** M3.8 makes the refusal clean; it does not
  make it a rate limit per client, and a determined caller can reconnect in a loop.
- **Prompt injection is mitigated, not prevented.**
- **Audio in memory is readable by anything with access to the process.** On a
  shared free instance that is a trust assumption about the host, not a control we
  operate.
