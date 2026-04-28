import re

CHAPTER_HEADERS = [
    "Mattie Michael",
    "Etta Mae Johnson",
    "Kiswana Browne",
    "Lucielia Louise Turner",
    "Cora Lee",
    "The Two",
]
SKIP_HEADERS = ["The Block"]
_ALL_HEADERS = CHAPTER_HEADERS + SKIP_HEADERS


def split_chapters(text):
    """Split novel text into {character: chapter_text}. Skips 'The Block' sections."""
    found = []
    for header in _ALL_HEADERS:
        pattern = re.compile(
            r"(?:^|\n)[ \t]*" + re.escape(header) + r"[ \t]*(?:\n|$)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            found.append((m.start(), m.end(), header))

    found.sort(key=lambda x: x[0])

    found_lower = {f[2].lower() for f in found}
    for h in CHAPTER_HEADERS:
        if h.lower() not in found_lower:
            print(f"WARNING: Chapter header not found in text: '{h}'")

    chunks = {}
    for i, (start, end, header) in enumerate(found):
        canonical = _canonical(header)
        if canonical is None:
            continue
        next_start = found[i + 1][0] if i + 1 < len(found) else len(text)
        chunk_text = text[end:next_start].strip()
        chunks[canonical] = chunk_text

    return chunks


def _canonical(header):
    """Return the canonical chapter name, or None if the header should be skipped."""
    for skip in SKIP_HEADERS:
        if header.lower() == skip.lower():
            return None
    for ch in CHAPTER_HEADERS:
        if header.lower() == ch.lower():
            return ch
    return None


def get_sentences_with_positions(chapter_text, nlp):
    """Return [(sentence_text, position_label)] for all sentences in a chapter.

    position_label is 'early', 'middle', or 'late' based on equal thirds by sentence index.
    """
    doc = nlp(chapter_text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    n = len(sentences)
    result = []
    for i, sent in enumerate(sentences):
        if i < n / 3:
            label = "early"
        elif i < 2 * n / 3:
            label = "middle"
        else:
            label = "late"
        result.append((sent, label))
    return result
