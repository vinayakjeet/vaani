from __future__ import annotations

from collections.abc import AsyncIterator

from vaani.sentences import MAX_BUFFER_CHARS, from_stream, sentences, split


async def stream(tokens: list[str], consumed: list[str] | None = None) -> AsyncIterator[str]:
    for token in tokens:
        if consumed is not None:
            consumed.append(token)
        yield token


async def collect(tokens: list[str]) -> list[str]:
    return [sentence async for sentence in from_stream(stream(tokens))]


def test_a_danda_ends_a_sentence() -> None:
    assert split("आपकी आय सीमा से कम है।", final=True) == (["आपकी आय सीमा से कम है।"], "")


def test_a_double_danda_ends_a_sentence() -> None:
    complete, tail = split("यह योजना अब बंद है॥ अगली बार देखिए।", final=True)

    assert complete == ["यह योजना अब बंद है॥", "अगली बार देखिए।"]
    assert tail == ""


def test_a_sentence_ending_in_an_amount_is_complete() -> None:
    """The most common shape of a scheme answer, and the one that broke.

    A full stop after a digit was read as part of the number whether or not
    anything followed it, so `Aapki limit hai 300000.` produced no sentence at
    all and synthesis waited for the whole reply. Devanagari hid it: the same
    sentence ending in `है।` terminates on the danda and never reaches the rule.
    """
    assert split("Aapki income limit hai 300000.", final=True) == (
        ["Aapki income limit hai 300000."],
        "",
    )


def test_a_decimal_point_is_not_a_boundary() -> None:
    assert split("Aapke paas 2.5 hectare zameen hai.", final=True) == (
        ["Aapke paas 2.5 hectare zameen hai."],
        "",
    )


def test_an_abbreviation_is_not_a_sentence_of_its_own() -> None:
    """`Rs.` is two characters and a full stop. Split there and synthesis says
    "rupees" alone, pauses for a network round trip, then says the amount."""
    assert split("Rs. 6000 saal mein milta hai.", final=True) == (
        ["Rs. 6000 saal mein milta hai."],
        "",
    )


def test_a_numbered_list_marker_is_not_a_boundary() -> None:
    assert split("Iske liye 2. Form bharna hoga.", final=True) == (
        ["Iske liye 2. Form bharna hoga."],
        "",
    )


def test_a_sentence_ending_in_an_amount_mid_reply_flushes_on_its_own() -> None:
    """The streaming half of the amount bug, and the more expensive half.

    Suppressing every full stop that follows a digit also suppresses this one, so
    sentence one waits for sentence two and the first-sentence flush buys nothing
    on the commonest shape of scheme answer. Only the length of the number
    separates it from a list marker.
    """
    complete, tail = split("Aapki limit hai 300000. Aap eligible hain.", final=False)

    assert complete == ["Aapki limit hai 300000.", "Aap eligible hain."]
    assert tail == ""


def test_a_year_ends_a_sentence() -> None:
    complete, _tail = split("Yeh scheme bandh hui 2019. Nayi wali dekhiye.", final=False)

    assert complete[0] == "Yeh scheme bandh hui 2019."


def test_a_grouped_amount_reads_as_one_number() -> None:
    """`1,50,000` is five digits with separators, not a three-digit number
    followed by a stray group, so it clears the amount threshold either way. The
    test exists because the digit run has to look past the commas to see that."""
    complete, _tail = split("Aapki aay 1,50,000. Aap eligible nahi hain.", final=False)

    assert complete[0] == "Aapki aay 1,50,000."


def test_hinglish_mixes_both_terminators_in_one_reply() -> None:
    complete, tail = split("Haan aap eligible hain. आपको 6000 मिलेगा।", final=True)

    assert complete == ["Haan aap eligible hain.", "आपको 6000 मिलेगा।"]
    assert tail == ""


def test_a_short_acknowledgement_merges_into_the_next_sentence() -> None:
    """Pins a deliberate latency cost rather than a bug.

    A reply opening with `हाँ।` could flush four characters of audio almost
    instantly, but synthesising it alone spends a round trip on two syllables
    and then leaves a gap before the real answer. It merges forward instead, so
    a reply that opens with an acknowledgement flushes later than one that does
    not. The ablation reports that, since it is the kind of trade a median hides.
    """
    assert sentences_of("हाँ। आप eligible हैं।") == ["हाँ। आप eligible हैं।"]


def test_an_unterminated_reply_still_yields() -> None:
    assert sentences_of("Aap eligible hain lekin") == ["Aap eligible hain lekin"]


def sentences_of(text: str) -> list[str]:
    return list(sentences(text))


async def test_the_first_sentence_is_yielded_before_the_stream_ends() -> None:
    """The acceptance criterion for M1.4, and it has to be about ordering.

    Asserting only the sentences would pass against an implementation that
    buffers the whole reply and splits it at the end, which is the M0 baseline
    this milestone exists to beat.
    """
    tokens = ["Aap ", "eligible ", "hain. ", "Aapko ", "6000 ", "milega."]
    consumed: list[str] = []
    first_seen_after = None

    async for _sentence in from_stream(stream(tokens, consumed)):
        first_seen_after = len(consumed)
        break

    assert first_seen_after is not None
    assert first_seen_after < len(tokens)


async def test_a_decimal_split_across_tokens_is_not_broken() -> None:
    """Mid-stream, a full stop after a digit is genuinely undecidable.

    The buffer ends at `2.` for as long as it takes the next token to arrive, and
    at that instant a decimal and a finished sentence look identical. Flushing
    then says "two point" and starts a new sentence at "5 lakh", which is the
    pause-mid-number failure the whole rule exists to prevent. So the decision
    waits for one more character, and only a stream that has ended treats a
    trailing digit-then-stop as a full stop.

    The prefix is long enough on purpose. An earlier version of this test used
    "Total 2." and passed against an implementation that split there anyway,
    because at eight characters `MIN_SENTENCE_CHARS` discarded the fragment
    before the rule was consulted. It was green for a reason unrelated to what it
    claimed to check.
    """
    assert await collect(["Aapke paas total ", "2", ".", "5", " hectare zameen hai."]) == [
        "Aapke paas total 2.5 hectare zameen hai."
    ]


async def test_a_stream_ending_on_an_amount_flushes_it() -> None:
    assert await collect(["Aapki limit hai ", "300000."]) == ["Aapki limit hai 300000."]


async def test_a_long_clause_with_no_terminator_flushes_rather_than_waiting() -> None:
    """A model that emits no terminator must not hold the audio forever. An
    awkward break is recoverable; silence that never ends is not."""
    clause = "haan " * (MAX_BUFFER_CHARS // 5 + 1)

    assert await collect([clause]) != []


async def test_an_empty_stream_yields_nothing() -> None:
    assert await collect([]) == []


async def test_whitespace_alone_is_not_a_sentence() -> None:
    assert await collect(["   ", "\n"]) == []
