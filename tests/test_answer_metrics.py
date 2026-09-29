from app.evaluation.answer_metrics import extract_citations, invalid_citations, is_refusal


def test_extract_citations_finds_every_bracket_number():
    assert extract_citations("Claim one [1]. Claim two [2][3].") == [1, 2, 3]


def test_extract_citations_of_no_citations_is_empty():
    assert extract_citations("No citations here.") == []


def test_invalid_citations_flags_out_of_range_numbers():
    assert invalid_citations("See [1] and [4].", num_chunks=3) == [4]


def test_invalid_citations_of_all_valid_numbers_is_empty():
    assert invalid_citations("See [1] and [2].", num_chunks=3) == []


def test_invalid_citations_flags_zero_and_negative_as_invalid():
    assert invalid_citations("See [0].", num_chunks=3) == [0]


def test_is_refusal_matches_the_exact_sentence():
    assert is_refusal("I don't have enough information in the provided documents.")


def test_is_refusal_rejects_a_close_but_different_sentence():
    assert not is_refusal("I don't have enough information in these documents.")