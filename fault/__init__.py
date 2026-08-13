"""Fault injection for tests that need a dependency to misbehave.

Every fault is served over HTTP, because everything this pipeline talks to is
HTTP: the OTLP exporter, the LLM providers, the transcriber and the synthesiser.
One server can therefore stand in for any of them, and a test picks the failure
rather than the mechanism.

Carried over from Spanlight rather than written again, per M3.1, and extended with
the shape Spanlight never needed: a fault that arrives after the response has
already started. Those are a different problem. A request that fails outright is
caught by any status check, while a stream that dies partway through has already
had its status checked, its headers read, and half its content spoken aloud. The
caller is committed by then, and the only question left is what it does with a
half-finished answer.

Three streaming shapes, and they are the ones that actually happen rather than a
taxonomy invented for completeness:

- `STREAM_DROP`, the connection closes mid-stream with no terminator. This is the
  dangerous one, because a truncated stream and a complete one look identical
  unless something checks for the terminator.
- `STREAM_STALL`, frames stop arriving and the connection stays open. Nothing
  errors and the caller simply waits, which is why a read timeout is not optional.
- `STREAM_MALFORMED_FRAME`, one unreadable frame in an otherwise healthy stream.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Fault(Enum):
    """How the far side fails.

    These are the shapes that actually show up in QUOTAS.md and in ShipGate's
    logs, not a taxonomy invented for completeness. `HANG` is the one worth
    naming: a dependency that is slow is far more dangerous than one that is
    down, because nothing reports an error and the caller simply stops.
    """

    UNREACHABLE = "unreachable"
    SERVER_ERROR = "server_error"
    HANG = "hang"
    RESET = "reset"
    RATE_LIMITED = "rate_limited"
    MALFORMED = "malformed"

    # Faults that arrive after the response has started. `NONE` is a healthy stream,
    # which every fault test needs as its control: a test that only ever sees the
    # failure cannot tell a working client from one that fails on everything.
    NONE = "none"
    STREAM_DROP = "stream_drop"
    STREAM_STALL = "stream_stall"
    STREAM_MALFORMED_FRAME = "stream_malformed_frame"


STREAMING_FAULTS = frozenset(
    {Fault.NONE, Fault.STREAM_DROP, Fault.STREAM_STALL, Fault.STREAM_MALFORMED_FRAME}
)

# Enough words to be a sentence, so a test can assert on what survived a fault.
DEFAULT_STREAM_WORDS = ("Aap ", "eligible ", "hain. ", "Aapko ", "6000 ", "milega.")


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 so the end of the body is the close of the connection. A streamed
    # response has no Content-Length to send, and this avoids hand-rolling chunked
    # encoding just to let a client see where the stream ended.
    protocol_version = "HTTP/1.0"

    def do_POST(self) -> None:
        try:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
        except Exception:
            return

        self.server.requests.append(self.path)
        fault = self.server.fault

        if fault in STREAMING_FAULTS:
            self._stream(fault)
            return

        if fault is Fault.HANG:
            # Outlives any sane client timeout without pinning the thread
            # forever, so a test that forgets to set one still finishes.
            time.sleep(self.server.hang_seconds)
            return
        if fault is Fault.RESET:
            self.close_connection = True
            self.wfile.close()
            return
        if fault is Fault.SERVER_ERROR:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"upstream is having a day")
            return
        if fault is Fault.RATE_LIMITED:
            self.send_response(429)
            self.send_header("Retry-After", str(self.server.retry_after))
            self.end_headers()
            self.wfile.write(b'{"error": "rate limit exceeded"}')
            return
        if fault is Fault.MALFORMED:
            # A 200 carrying nonsense. Worse than an error status, because every
            # status check passes and the failure surfaces at parse time.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{not json at all")
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"")

    def _stream(self, fault: Fault) -> None:
        """Serve server-sent events, then misbehave partway through."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        words = self.server.stream_words
        break_at = self.server.frames_before_fault

        for index, word in enumerate(words):
            if index == break_at and fault is not Fault.NONE:
                if fault is Fault.STREAM_DROP:
                    # No terminator, no closing frame, just gone. The client has
                    # already had a 200 and half a sentence.
                    self.close_connection = True
                    return
                if fault is Fault.STREAM_STALL:
                    # Held open and silent. Outlives any sane read timeout without
                    # pinning the thread forever, so a test that forgets to set one
                    # still finishes.
                    time.sleep(self.server.hang_seconds)
                    return
                if fault is Fault.STREAM_MALFORMED_FRAME:
                    self._write(b"data: {not json at all\n\n")

            self._write(_sse_content(word))

        self._write(b"data: [DONE]\n\n")

    def _write(self, payload: bytes) -> None:
        """Write one frame, so the client sees it as it is produced.

        No flush. `wfile` on this handler is an unbuffered socket writer, so a flush
        is a no-op: adding one and then deleting it again changed no test, which is
        the same evidence that removed Spanlight's `atexit` hook. Code that looks
        load-bearing and is not costs a reader more than it saves.

        `test_frames_arrive_before_the_stream_ends` still pins the property, since it
        is the property that matters rather than the mechanism, and it would go red if
        a future change started buffering.
        """
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # The client abandoned the stream, which several tests do on purpose.
            self.close_connection = True

    def log_message(self, *args: object) -> None:
        pass


class _Server(ThreadingHTTPServer):
    """Threaded, with daemon threads, specifically because of `HANG`.

    A single-threaded server handles one request at a time in the accept loop, so
    a handler sleeping out a hang blocks `shutdown()` for the full duration. The
    suite then pays the hang twice: once for the client timeout it is testing,
    and again waiting to tear the server down.
    """

    daemon_threads = True

    fault: Fault
    requests: list[str]
    hang_seconds: float
    retry_after: int
    stream_words: tuple[str, ...]
    frames_before_fault: int


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def faulty_endpoint(
    fault: Fault,
    hang_seconds: float = 30.0,
    retry_after: int = 40,
    stream_words: tuple[str, ...] = DEFAULT_STREAM_WORDS,
    frames_before_fault: int = 2,
) -> Iterator[_Server]:
    """Serve `fault` at a base URL, yielding the server so a test can count hits.

    `UNREACHABLE` binds nothing and hands back a port with no listener, which is
    a connection refused rather than a simulated one.

    `retry_after` defaults to 40 seconds because that is what a real Gemini 429
    asked for, recorded in QUOTAS.md. Five attempts of exponential backoff total
    about 31, so a client that ignores the header retries entirely inside the
    cooldown and fails with quota to spare.
    """
    port = _free_port()

    if fault is Fault.UNREACHABLE:
        server = _Server.__new__(_Server)
        server.requests = []
        server.url = f"http://127.0.0.1:{port}"
        yield server
        return

    server = _Server(("127.0.0.1", port), _Handler)
    server.fault = fault
    server.requests = []
    server.hang_seconds = hang_seconds
    server.retry_after = retry_after
    server.stream_words = stream_words
    server.frames_before_fault = frames_before_fault
    server.url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _sse_content(text: str) -> bytes:
    """One OpenAI-shaped delta frame, as a provider writes it."""
    return b"data: " + json.dumps({"choices": [{"delta": {"content": text}}]}).encode() + b"\n\n"
