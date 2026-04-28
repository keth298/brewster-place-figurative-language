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


def _header_patterns(header):
    """Return compiled regex patterns that match a chapter header (single-line and split-line)."""
    single = re.compile(
        r"(?:^|\n)[ \t\x0c\r]*" + re.escape(header) + r"[ \t\x0c\r]*(?:\n|$)",
        re.IGNORECASE,
    )
    patterns = [single]
    words = header.split()
    if len(words) > 1:
        first = re.escape(" ".join(words[:-1]))
        last = re.escape(words[-1])
        split = re.compile(
            r"(?:^|\n)[ \t\x0c\r]*" + first + r"[ \t\x0c\r]*\n[ \t\x0c\r]*" + last + r"[ \t\x0c\r]*(?:\n|$)",
            re.IGNORECASE,
        )
        patterns.append(split)
    return patterns


def split_chapters(text):
    """Split novel text into {character: chapter_text}. Skips 'The Block' sections."""
    found = []
    seen_positions = set()
    for header in _ALL_HEADERS:
        for pattern in _header_patterns(header):
            for m in pattern.finditer(text):
                if m.start() not in seen_positions:
                    seen_positions.add(m.start())
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
