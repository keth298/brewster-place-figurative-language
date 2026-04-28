import re

_LIKE_AS_RE = re.compile(r"\b(like|as)\b", re.IGNORECASE)

_FALSE_POS_AS = [
    re.compile(r"\bas\s+soon\s+as\b", re.IGNORECASE),
    re.compile(r"\bas\s+well\b", re.IGNORECASE),
    re.compile(r"\bas\s+long\s+as\b", re.IGNORECASE),
    re.compile(r"\bas\s+if\b", re.IGNORECASE),
    re.compile(r"\bas\s+though\b", re.IGNORECASE),
]
_FALSE_POS_LIKE = [
    re.compile(r"\bjust\s+like\s+that\b", re.IGNORECASE),
]

_VALID_DEPS = {"prep", "mark", "advcl"}


def _is_false_positive(token, sent_text):
    word = token.text.lower()
    if word == "like":
        if token.pos_ == "VERB":
            return True
        for pat in _FALSE_POS_LIKE:
            if pat.search(sent_text):
                return True
    if word == "as":
        for pat in _FALSE_POS_AS:
            if pat.search(sent_text):
                return True
    return False


def _has_comparative_structure(token):
    """Return True if token links two noun phrases via a comparative dependency."""
    if token.dep_ not in _VALID_DEPS:
        return False
    head = token.head
    if head.pos_ not in {"VERB", "NOUN", "ADJ", "AUX"}:
        return False
    has_np = any(
        t.pos_ in {"NOUN", "PRON", "PROPN"}
        for t in token.subtree
        if t != token
    )
    return has_np


def detect(chunks, nlp):
    """Detect similes in chunks.

    chunks: {character: [(sentence_text, position_label), ...]}
    Returns list of dicts with keys: character, sentence, type, position.
    """
    results = []
    for character, sentences in chunks.items():
        for sent_text, position in sentences:
            if not _LIKE_AS_RE.search(sent_text):
                continue
            doc = nlp(sent_text)
            for token in doc:
                if token.text.lower() not in ("like", "as"):
                    continue
                if _is_false_positive(token, sent_text):
                    continue
                if _has_comparative_structure(token):
                    results.append({
                        "character": character,
                        "sentence": sent_text,
                        "type": "simile",
                        "position": position,
                    })
                    break  # at most one simile entry per sentence
    return results
