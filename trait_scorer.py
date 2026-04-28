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

_sia = SentimentIntensityAnalyzer()


def score_keywords(text, keywords):
    if not text or not text.split():
        return 0.0
    pattern = r'\b(?:' + '|'.join(re.escape(w) for w in keywords) + r')\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    count = len(matches)
    if count == 0:
        return 0.0
    return count / len(text.split()) * 1000


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
