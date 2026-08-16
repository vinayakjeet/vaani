"""M2.4's regression assertion. The bench publishes the numbers; this stops them moving.

Nothing here is a target. The 100ms figure in the tail-control brief is a budget to
verify against once there is a measurement, and this portfolio has twice paid for the
other order: a gate set at 2 points against a judge whose measured noise floor was 20,
and a detector threshold that fired on a pattern already written down as healthy. So the
bounds below are the structural floor each number cannot go under, plus headroom, and
the floor is a constant the system already declares rather than a number chosen here.

**What each bound is made of.** An interruption cannot be noticed before the detector
that notices it has enough evidence, and each of the three numbers has a different such
floor: three frames for the browser's own level detector, `min_speech_ms` for the
server's endpointer, `commit_ms` for the duration path. What can regress is the overhead
above that floor, which is this pipeline's own cost, and that is what is bounded. A bound
on the total would pass forever by being mostly threshold.

**Nothing here is asserted in milliseconds of wall clock.** Every threshold in this
pipeline is counted in frames of audio, so the earliest moment a decision can be reached
is the moment its nth frame arrived, and this harness paces frames with a `sleep` rather
than a sound card. Measured at 20.7ms a frame on an idle machine and 28.0ms under a full
suite, which moved the headline pause from 62ms to 84ms with no code change at all. So
each assertion is against the frame the harness actually sent, and what is left over is
this pipeline's own cost.

`OVERHEAD_BUDGET_MS` is the one number here without a derivation, and it is headroom
rather than a claim. The overheads it bounds were measured at under 3ms across every arm
once pacing was taken out, so the bound is fifty times the observed worst case: it cannot
catch a regression of a few milliseconds, and it will catch cancellation growing a wait,
which is the failure that would actually happen. `bench/bargein.py --runs 20` is where
the real distribution is published.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bench.bargein import (
    ARMS,
    PREEMPT_FRAMES,
    PREEMPT_THRESHOLD_RMS,
    Run,
    measure,
    summarise,
)
from vaani.endpoint import DEFAULT_MIN_SPEECH_MS, DEFAULT_TRAILING_SILENCE_MS
from vaani.protocol import FRAME_MS
from vaani.turn_taking import DEFAULT_COMMIT_MS

# Everything above the floor a detector cannot beat. Headroom for a shared runner, not a
# target, and the measured value it is headroom over is in this module's docstring.
OVERHEAD_BUDGET_MS = 150

# Two per arm, because the assertions below are about a floor being cleared rather than
# about a distribution, and this suite pays for every run in real time: the frames are
# paced at 20ms deliberately, so a bench that ran faster would be measuring itself.
RUNS = 2

PAGE = Path(__file__).resolve().parents[2] / "web" / "index.html"


@pytest.fixture(scope="module")
async def runs() -> list[Run]:
    """One set of interruptions, shared. Driving the whole pipeline three times per
    assertion would put a minute of real-time audio pacing into this file."""
    return await measure(RUNS)


def by_arm(runs: list[Run], name: str) -> list[Run]:
    return [run for run in runs if run.arm == name]


async def test_the_backdated_start_matches_the_frame_the_speech_began_on(runs) -> None:
    """The measurement's own correctness, and the only place it can be checked.

    Every barge-in number the deployed system reports is measured from a start it
    infers: detection cannot fire until `min_speech_ms` of speech has arrived, so the
    clock is wound back by the length of the speech run to reach the frame it began on.
    Nothing in production can tell whether that inference is right, because production
    has no record of which frame that was. The harness sent it, so it does.

    Started at detection instead, every number here would lose 200ms in the flattering
    direction, and the code would look correct while doing it.

    Net of the harness's own pacing. The session counts frames and assumes each is 20ms,
    which a sound card guarantees and a `sleep` loop does not, so a busy machine puts its
    own slop into this number nine times over and fails a test about something else. The
    raw disagreement is published in the bench report; what is asserted here is the part
    the session is answerable for.
    """
    errors = [
        abs(run.clock_error_net_ms) for run in runs if run.clock_error_net_ms is not None
    ]

    assert errors, "no interruption produced a clock to check"
    assert max(errors) < FRAME_MS, (
        f"the session's backdated start is off by {max(errors):.1f}ms, more than the "
        f"{FRAME_MS}ms frame it is inferred from"
    )


async def test_the_client_stops_the_audio_before_the_server_asks_it_to(runs) -> None:
    """What M2.11 buys, as a number rather than as an argument.

    The browser pauses on its own level detector after three frames; the server needs
    `min_speech_ms` before its endpointer will say anything at all. So the client is
    ahead by the difference before a network is involved at all, and on the deployed
    stack it is further ahead by a one-way trip. If this ever inverts, the client-side
    preemption has stopped working and the only symptom is a slower barge-in.
    """
    for run in runs:
        assert run.client_paused_ms is not None
        assert run.server_paused_ms is not None
        assert run.client_paused_ms < run.server_paused_ms
        assert run.paused_ms == run.client_paused_ms


async def test_each_pause_clears_its_own_detectors_floor_and_little_else(runs) -> None:
    """Both halves, against the frame of audio each one needs before it can fire.

    Against the frame that arrived rather than the frame that should have. Every
    threshold here is counted in frames, so on a machine feeding audio at 28ms a frame
    the earliest possible pause moves with it, and asserting against a nominal 20ms would
    fail whenever the suite is busy for a reason that has nothing to do with this code.
    """
    for run in runs:
        client = run.overhead_ms(run.client_paused_ms, PREEMPT_FRAMES)
        server = run.overhead_ms(run.server_paused_ms, DEFAULT_MIN_SPEECH_MS // FRAME_MS)

        assert client is not None and server is not None
        assert 0 <= client < OVERHEAD_BUDGET_MS
        assert 0 <= server < OVERHEAD_BUDGET_MS


async def test_abandoning_the_turn_costs_the_threshold_and_not_much_more(runs) -> None:
    """The duration path, where nothing is asked of a provider.

    `commit_ms` of sustained speech is the whole of the wait by construction, so what is
    left is cancelling the generation and the synthesis. That is the part a change can
    make slow: `SpeakingTurn.cancel` awaits the producer rather than only signalling it,
    and something added inside that await lands here.
    """
    for run in by_arm(runs, "duration"):
        overhead = run.overhead_ms(run.committed_ms, DEFAULT_COMMIT_MS // FRAME_MS)

        assert run.committed_ms is not None
        assert overhead is not None
        assert 0 <= overhead < OVERHEAD_BUDGET_MS


async def test_the_verified_path_costs_the_utterance_and_the_endpoint_wait(runs) -> None:
    """The other path, and the reason both are published separately.

    Here the interruption is short, so it is not certain, and it cannot be resolved until
    the utterance has ended: 400ms of speech plus the trailing silence the endpointer
    waits out. That is most of a second before anything is decided, against 800ms of
    speech on the duration path, and a median over both would be a median over two
    distributions.

    The verifier answers instantly in this harness and a recogniser does not. That term
    is stated as zero in the report rather than hidden, and `bench/stages.md` keeps the
    transcription inside this number on the deployed stack because the user waits through
    it either way.
    """
    arms = {arm.name: arm for arm in ARMS}

    for arm_name in ("verified_interruption", "verified_backchannel"):
        arm = arms[arm_name]
        # The utterance, then the quiet that ends it. Both counted in frames, because
        # the endpointer counts frames, and both sent by this harness so the moment they
        # landed is known rather than assumed.
        earliest = arm.interrupting_frames + DEFAULT_TRAILING_SILENCE_MS // FRAME_MS

        for run in by_arm(runs, arm_name):
            settled = run.committed_ms if run.committed_ms is not None else run.resumed_ms
            assert settled is not None
            assert settled > run.frame_time_ms(earliest)
            assert run.overhead_ms(settled, earliest) < OVERHEAD_BUDGET_MS


async def test_every_interruption_reaches_an_outcome(runs) -> None:
    """The guard on all of the above. A suspicion nobody resolves has no end boundary, so
    it is in no median, and a run of them would improve every number in the report by
    quietly removing the cases that went wrong."""
    report = summarise(runs)

    for name, arm in report["arms"].items():
        assert arm["n"] == RUNS, f"{name} produced {arm['n']} runs, expected {RUNS}"
        assert arm["unresolved"] == 0, f"{name} left {arm['unresolved']} interruptions open"


def test_the_bench_models_the_preemption_rule_the_page_actually_runs() -> None:
    """The bench duplicates the browser's level detector, because the client's own pause
    happens in the browser and is never reported, so the earlier of the two ends can only
    be taken somewhere that can see both. Duplication has a cost and this is it paid:
    change the page and the bench measures a client that no longer exists, silently, in
    the direction of whichever number the page moved."""
    page = PAGE.read_text(encoding="utf-8")

    frames = re.search(r"const PREEMPT_FRAMES = (\d+);", page)
    threshold = re.search(r"const THRESHOLD = (\d+);", page)

    assert frames is not None and threshold is not None, "the page's rule has been renamed"
    assert int(frames.group(1)) == PREEMPT_FRAMES
    assert int(threshold.group(1)) == PREEMPT_THRESHOLD_RMS


def test_the_arms_cover_both_paths_the_definitions_split_on() -> None:
    """`bench/stages.md` publishes the duration path and the verified path separately,
    on the grounds that they are different distributions. A bench that measured one of
    them would satisfy every assertion above and publish half the finding."""
    ends = {end for arm in ARMS for end in arm.ends}

    assert ends == {"paused", "committed", "resumed"}
    assert {arm.backchannel for arm in ARMS} == {None, True, False}
