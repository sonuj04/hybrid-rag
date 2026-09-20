from app.evaluation.typos import add_typos


def test_same_seed_gives_same_output():
    text = "How does neuroplasticity relate to depression"
    assert add_typos(text, seed=1) == add_typos(text, seed=1)


def test_short_words_are_never_changed():
    assert add_typos("how is it so", seed=1, rate=1.0) == "how is it so"


def test_long_words_change_but_keep_first_and_last_letter():
    text = "neuroplasticity hippocampus antidepressants"
    changed = add_typos(text, seed=3, rate=1.0)
    assert changed != text
    for original, new in zip(text.split(), changed.split()):
        assert original[0] == new[0]
        assert original[-1] == new[-1]