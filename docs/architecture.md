# Architecture

What is actually deployed, not an aspirational diagram. Regenerate this by hand
when a stage changes; there is no build step that keeps it honest automatically.

## One turn, overlapped

```mermaid
sequenceDiagram
    participant Browser
    participant Socket as WebSocket<br/>(app/routers/voice.py)
    participant Session as VoiceSession<br/>(vaani/session.py)
    participant Pipeline as StreamingPipeline<br/>(vaani/pipeline.py)
    participant STT as STT stream<br/>(Groq Whisper, chunked)
    participant LLM as LLM turn<br/>(Groq, tool-calling)
    participant TTS as TTS stream<br/>(EdgeTts, per sentence)

    Browser->>Socket: PCM16 frames, 20ms each
    Socket->>Session: frame
    Session->>Session: Endpointer.accept(frame)
    Note over Session: trailing silence times out,<br/>or a partial already looks finished
    Session->>Pipeline: run(frames)
    Pipeline->>STT: frames, incremental
    STT-->>Pipeline: partials, then final transcript
    Pipeline->>Pipeline: numerals.normalise, confirm.needs_confirming
    Pipeline->>LLM: transcript + history
    LLM-->>Pipeline: tool call (check_eligibility / find_schemes)
    Pipeline->>Pipeline: tools.dispatch (fixture data, never invented)
    LLM-->>Pipeline: reply text, streamed
    Pipeline->>Pipeline: sentences.from_stream (segment as it arrives)
    Pipeline->>Pipeline: grounding.check (every number sourced or refused)
    Pipeline->>TTS: sentence 1
    TTS-->>Session: audio chunk
    Session->>Socket: audio_start, then chunks
    Socket->>Browser: audio bytes
    Note over Pipeline,TTS: sentence 2 synthesises while<br/>sentence 1 is still playing
```

Sentence one's audio reaches the browser before the model has finished writing
sentence three. That overlap, not any single provider being fast, is what M1
built and M5 measures the size of.

## Barge-in

```mermaid
sequenceDiagram
    participant Browser
    participant Session as VoiceSession
    participant Speaking as SpeakingTurn<br/>(vaani/barge_in.py)

    Note over Session,Speaking: agent is speaking, generation N
    Browser->>Session: speech detected, mid-playback
    Session->>Session: TurnTaking: duration or verified?
    alt short, ambiguous
        Session->>Speaking: pause (WAIT state)
        Session->>Session: transcribe the interrupting audio
        alt backchannel
            Session->>Speaking: resume
        else genuine interruption
            Session->>Session: generation N+1, truncate history at word played
            Session->>Speaking: cancel()
        end
    else long enough to commit
        Session->>Session: generation N+1
        Session->>Speaking: cancel()
    end
    Speaking->>Speaking: aclose() the answer generator,<br/>in the task that opened it
    Session->>Browser: pause / resume
```

`SpeakingTurn.cancel()` awaits the cancelled task rather than firing and
forgetting: a cancel that returns early leaves a provider connection open
against a quota nobody is watching.

## Deployed today

```mermaid
flowchart LR
    subgraph Browser
        Mic[getUserMedia] --> Worklet[AudioWorklet<br/>PCM16, 20ms frames]
    end
    Worklet <-->|WebSocket| Backend

    subgraph Backend[Render, free tier]
        Router[FastAPI<br/>app/routers/voice.py]
        Session[VoiceSession]
        Pipeline[StreamingPipeline]
        Router --> Session --> Pipeline
    end

    Pipeline -->|whisper-large-v3-turbo| Groq1[Groq: STT]
    Pipeline -->|openai/gpt-oss-120b, low reasoning| Groq2[Groq: LLM]
    Pipeline -->|hi-IN-SwaraNeural| Edge[EdgeTts: TTS]

    Backend -.->|OTLP, when configured| Grafana[Grafana Cloud<br/>Spanlight traces]

    Pipeline -.->|bench-only, credit-limited| Sarvam1[Sarvam: Saaras STT]
    Pipeline -.->|bench-only, credit-limited| Sarvam2[Sarvam: Bulbul TTS]
```

Solid edges run on every real turn. Dashed edges are measurement paths: Sarvam
is `bench/waterfall.py --stack sarvam` only, never the deployed service
(QUOTAS.md), and Grafana Cloud traces are emitted whenever
`OTEL_EXPORTER_OTLP_ENDPOINT` is set and silently skipped when it is not,
logged once at startup either way.

The frontend is two static files on GitHub Pages
(`web/index.html`, `web/capture-worklet.js`), not the Vercel deployment SPEC
originally named; recorded as a deviation in DECISIONS.md.
