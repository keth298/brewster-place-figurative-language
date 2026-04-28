import re
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

RACIAL_KEYWORDS = {
    "black", "african", "afro", "revolution", "community", "movement",
    "white", "people", "race", "struggle", "protest", "liberation",
    "heritage", "identity"
}

DOMESTIC_KEYWORDS = {
    "husband", "baby", "child", "children", "home", "clean", "cook",
    "dinner", "mother", "marriage", "pregnant", "eugene", "basil"
}

_RACIAL_PATTERN = re.compile(r'\b(?:' + '|'.join(sorted(RACIAL_KEYWORDS)) + r')\b', re.IGNORECASE)
_DOMESTIC_PATTERN = re.compile(r'\b(?:' + '|'.join(sorted(DOMESTIC_KEYWORDS)) + r')\b', re.IGNORECASE)

_COMPILED = {
    frozenset(RACIAL_KEYWORDS): _RACIAL_PATTERN,
    frozenset(DOMESTIC_KEYWORDS): _DOMESTIC_PATTERN,
}

_sia = SentimentIntensityAnalyzer()


def score_keywords(text, keywords):
    words = text.split()
    if not words:
        return 0.0
    pattern = _COMPILED.get(frozenset(keywords))
    if pattern is None:
        pattern = re.compile(r'\b(?:' + '|'.join(sorted(keywords)) + r')\b', re.IGNORECASE)
    count = len(pattern.findall(text))
    return count / len(words) * 1000


def score_sentiment(sentences):
    if not sentences:
        return 0.0
    scores = [_sia.polarity_scores(s)["compound"] for s in sentences]
    return sum(scores) / len(scores)


def score_character(chapter_text, sentences):
    return {
        "racial_consciousness": score_keywords(chapter_text, RACIAL_KEYWORDS),
        "emotional_register": score_sentiment(sentences),
        "domestic_score": score_keywords(chapter_text, DOMESTIC_KEYWORDS),
    }
