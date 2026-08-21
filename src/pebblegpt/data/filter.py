"""Document-level quality filtering.

The n-gram repetition filter follows OLMo 2, which found that removing
documents with 32+ repetitions of 1-13 token spans significantly reduced
loss-spike frequency during training.

Note on n=1: OLMo 2's threshold is defined over *model* tokens. Applied to
whitespace words, unigrams reject far too much — common words like "the"
legitimately appear 32+ times in any document over a few hundred words. We
start at n=2 and scan a capped prefix at representative n values, which keeps
the filter fast and stops it discarding ~40% of already-clean corpora.
"""

from collections import Counter

MAX_REPEATS = 32
MAX_WORDS_TO_SCAN = 10_000
NGRAM_SIZES = (2, 3, 5, 8, 13)
MIN_CHARS = 50


def has_excessive_repetition(text: str, max_repeats: int = MAX_REPEATS) -> bool:
    """True if any n-gram repeats more than max_repeats times."""
    words = text.split()[:MAX_WORDS_TO_SCAN]
    if not words:
        return False

    for n in NGRAM_SIZES:
        if len(words) < n * 2:
            break
        ngrams = Counter(
            tuple(words[i:i + n]) for i in range(len(words) - n + 1)
        )
        if ngrams and ngrams.most_common(1)[0][1] > max_repeats:
            return True

    return False


def is_too_short(text: str, min_chars: int = MIN_CHARS) -> bool:
    return len(text.strip()) < min_chars


def keep_document(text: str) -> bool:
    """Master filter. Returns True if the document should be trained on."""
    if not text or is_too_short(text):
        return False
    if has_excessive_repetition(text):
        return False
    return True