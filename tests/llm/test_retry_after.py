from __future__ import annotations

import httpx
import pytest

from llm.providers.base import parse_retry_after, total_aware_usage_parser


def response(status: int = 429, json_body=None, headers=None) -> httpx.Response:
    return httpx.Response(status, json=json_body, headers=headers or {})


def test_prefers_the_standard_header():
    resp = response(headers={"Retry-After": "30"}, json_body={"error": {"message": "retry in 9s"}})
    assert parse_retry_after(resp) == 30.0


def test_reads_google_retry_info_detail():
    resp = response(
        json_body={
            "error": {
                "message": "quota exceeded",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "48s"},
                ],
            }
        }
    )
    assert parse_retry_after(resp) == 48.0


def test_falls_back_to_the_message_text():
    """The real shape Gemini returned: no header, no RetryInfo, delay only in
    prose. Without this the throttle waits its short default and hammers a limit
    that needs the better part of a minute."""
    resp = response(
        json_body=[
            {
                "error": {
                    "code": 429,
                    "message": (
                        "You exceeded your current quota. "
                        "* Quota exceeded for metric: "
                        "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                        "limit: 20, model: gemini-3.6-flash\n"
                        "Please retry in 48.63971551s."
                    ),
                    "status": "RESOURCE_EXHAUSTED",
                }
            }
        ]
    )
    assert parse_retry_after(resp) == pytest.approx(48.6397, abs=1e-3)


def test_handles_the_list_wrapped_error_object():
    """Gemini wraps its error in a single-element list, which is not the OpenAI
    shape the rest of the client assumes."""
    resp = response(json_body=[{"error": {"message": "Please retry in 12s."}}])
    assert parse_retry_after(resp) == 12.0


def test_millisecond_delays_are_not_read_as_seconds():
    """Gemini reports sub-second waits in ms. Reading "607.269104ms" as 607
    seconds stalls a run for ten minutes over a delay of half a second, which is
    exactly what happened before this was handled."""
    resp = response(json_body=[{"error": {"message": "Please retry in 607.269104ms."}}])
    assert parse_retry_after(resp) == pytest.approx(0.607, abs=1e-3)


def test_second_delays_still_read_as_seconds():
    resp = response(json_body={"error": {"message": "Please retry in 39.5s."}})
    assert parse_retry_after(resp) == pytest.approx(39.5)


def test_returns_none_when_nothing_says():
    assert parse_retry_after(response(json_body={"error": {"message": "slow down"}})) is None


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        [],
        {"error": None},
        # A bare string rather than an object, which several providers return and
        # which has no fields to interrogate.
        {"error": "rate limited"},
        {"error": {"details": [{"@type": "x"}]}},
        {"error": {"details": "not a list"}},
        "not json at all",
    ],
)
def test_malformed_bodies_do_not_raise(body):
    """A rate limit response is already a bad moment. Parsing it must not turn a
    recoverable 429 into an unhandled exception."""
    resp = httpx.Response(429, json=body) if body != "not json at all" else httpx.Response(
        429, text="not json at all"
    )
    assert parse_retry_after(resp) is None


def test_bad_header_value_falls_through_to_the_body():
    resp = response(
        headers={"Retry-After": "soon"}, json_body={"error": {"message": "retry in 7s"}}
    )
    assert parse_retry_after(resp) == 7.0


# Token accounting, the other thing Gemini reports differently.


def test_total_aware_parser_captures_reasoning_tokens():
    """A real Gemini response: prompt 2, completion 9, total 197. The missing 186
    are reasoning tokens, and reading only the itemised fields undercounts the run
    by roughly eighteen times."""
    tokens_in, tokens_out = total_aware_usage_parser(
        {"usage": {"prompt_tokens": 2, "completion_tokens": 9, "total_tokens": 197}}
    )
    assert tokens_in == 2
    assert tokens_out == 195


def test_total_aware_parser_leaves_consistent_usage_alone():
    """When the numbers already add up there is nothing hidden, so the reported
    completion count is used as-is."""
    assert total_aware_usage_parser(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    ) == (10, 5)


def test_total_aware_parser_handles_missing_fields():
    assert total_aware_usage_parser({}) == (None, None)
    assert total_aware_usage_parser({"usage": {"prompt_tokens": 4}}) == (4, None)
