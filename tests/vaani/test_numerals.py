from vaani.numerals import normalise


def test_a_scale_word_multiplies_the_count_before_it():
    assert normalise("meri aay pachaas hazaar hai").text == "meri aay 50000 hai"


def test_the_indian_grouping_opens_a_new_section_at_lakh():
    # Folding multipliers left to right would give 1x100000x50x1000. The section
    # boundary is what makes this 150000, and it is the case a naive implementation
    # gets wrong silently.
    assert normalise("ek lakh pachaas hazaar").numbers == (150000,)


def test_hundreds_take_an_addition_and_thousands_do_not():
    assert normalise("do sau pachaas").numbers == (250,)


def test_two_adjacent_units_are_refused_rather_than_added():
    # "paanch pachaas" is not 55. Hindi says 55 as one word, so this is either a
    # misrecognition or a correction, and both want a confirmation turn.
    result = normalise("paanch pachaas rupaye")
    assert result.numbers == ()
    assert result.unresolved == ("paanch pachaas",)
    assert result.text == "paanch pachaas rupaye"
    assert not result.confident


def test_a_quarter_prefix_implies_one_before_a_scale():
    assert normalise("sava crore").numbers == (12500000,)


def test_a_quarter_prefix_applies_to_the_number_that_follows_it():
    assert normalise("sava do lakh").numbers == (225000,)
    assert normalise("paune do lakh").numbers == (175000,)


def test_a_fraction_word_is_a_value_in_its_own_right():
    assert normalise("dedh lakh").numbers == (150000,)
    assert normalise("dhai sau").numbers == (250,)


def test_digits_from_the_recogniser_still_take_a_spoken_scale():
    # Whisper routinely writes the count as digits and the scale as a word. Treating
    # those as unrelated tokens leaves the multiplier for the model to apply.
    assert normalise("50 hazaar").numbers == (50000,)
    assert normalise("2.5 lakh").numbers == (250000,)


def test_devanagari_digits_and_words_both_resolve():
    assert normalise("पचास हजार").numbers == (50000,)
    assert normalise("५० हजार").numbers == (50000,)


def test_a_non_integer_survives_as_a_decimal():
    assert normalise("dhai acre zameen").text == "2.5 acre zameen"


def test_trailing_punctuation_is_kept_so_sentence_splitting_still_works():
    assert normalise("meri aay pachaas hazaar.").text == "meri aay 50000."


def test_an_unknown_number_word_leaves_the_whole_run_alone():
    # "pachpan" is not in the lexicon. Normalising "pachpan hazaar" to "hazaar" or
    # to 1000 would both be worse than leaving it for a human to hear about.
    result = normalise("pachpan hazaar")
    assert result.text == "pachpan hazaar"
    assert result.unresolved == ("hazaar",)


def test_text_with_no_numbers_is_returned_unchanged():
    result = normalise("mujhe ghar chahiye")
    assert result.text == "mujhe ghar chahiye"
    assert result.numbers == ()
    assert result.confident


def test_an_offset_with_nothing_after_it_is_refused():
    assert normalise("sava").unresolved == ("sava",)


def test_a_bare_amount_in_digits_is_left_as_it_is():
    result = normalise("300000")
    assert result.numbers == (300000,)
    assert result.text == "300000"
