"""Adds realistic typos to a question, to test how robust retrieval is to misspellings."""
import random

PUNCTUATION = ".,?!;:()\"'"


def _corrupt(core: str, rng: random.Random) -> str:
    """Swap two neighbouring letters or drop one letter. First and last letters stay put."""
    if rng.random() < 0.5:
        i = rng.randint(1, len(core) - 3)
        return core[:i] + core[i + 1] + core[i] + core[i + 2:]
    i = rng.randint(1, len(core) - 2)
    return core[:i] + core[i + 1:]


def add_typos(text: str, seed: int = 0, rate: float = 0.3, min_len: int = 5) -> str:
    """Corrupt roughly `rate` of the words that have at least `min_len` letters.

    The same seed always gives the same result. If no word was changed, the longest word is.
    """
    rng = random.Random(seed)
    words = []
    for word in text.split(" "):
        core = word.strip(PUNCTUATION)
        if len(core) >= min_len and rng.random() < rate:
            word = word.replace(core, _corrupt(core, rng), 1)
        words.append(word)
    result = " ".join(words)

    if result == text:
        long_words = [w for w in text.split(" ") if len(w.strip(PUNCTUATION)) >= min_len]
        if long_words:
            target = max(long_words, key=lambda w: len(w.strip(PUNCTUATION)))
            core = target.strip(PUNCTUATION)
            result = text.replace(core, _corrupt(core, rng), 1)
    return result