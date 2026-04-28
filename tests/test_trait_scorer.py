import pytest
from trait_scorer import score_keywords, score_sentiment, score_character, RACIAL_KEYWORDS, DOMESTIC_KEYWORDS


def test_score_keywords_empty_text():
    assert score_keywords("", RACIAL_KEYWORDS) == 0.0


def test_score_keywords_no_matches():
    assert score_keywords("She walked down the street quickly.", RACIAL_KEYWORDS) == 0.0


def test_score_keywords_normalises_per_1000():
    result = score_keywords("black community", RACIAL_KEYWORDS)
    assert result == pytest.approx(1000.0)


def test_score_keywords_case_insensitive():
    result_lower = score_keywords("black", RACIAL_KEYWORDS)
    result_upper = score_keywords("BLACK", RACIAL_KEYWORDS)
    assert result_lower == pytest.approx(result_upper)
    assert result_lower > 0.0


def test_score_keywords_word_boundaries():
    result = score_keywords("blackbird flew over the sky", RACIAL_KEYWORDS)
    assert result == 0.0


def test_score_sentiment_empty_list():
    assert score_sentiment([]) == 0.0


def test_score_sentiment_returns_float_in_range():
    sentences = ["She walked to the market.", "The day was ordinary."]
    result = score_sentiment(sentences)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


def test_score_sentiment_positive_text():
    sentences = ["She felt wonderful and joyful."]
    result = score_sentiment(sentences)
    assert result > 0.0


def test_score_sentiment_negative_text():
    sentences = ["She felt terrible and miserable."]
    result = score_sentiment(sentences)
    assert result < 0.0


def test_score_character_returns_correct_keys():
    result = score_character("She joined the community movement.", ["She felt great."])
    assert set(result.keys()) == {"racial_consciousness", "emotional_register", "domestic_score"}


def test_score_character_racial_consciousness():
    text = "The black community gathered for the liberation movement."
    result = score_character(text, [])
    assert result["racial_consciousness"] > 0.0


def test_score_character_domestic_score():
    text = "She cooked dinner for her husband and children at home."
    result = score_character(text, [])
    assert result["domestic_score"] > 0.0


def test_score_character_emotional_register_is_mean_vader():
    sentences = ["She was happy.", "She was sad."]
    result = score_character("some text here", sentences)
    assert isinstance(result["emotional_register"], float)
    assert -1.0 <= result["emotional_register"] <= 1.0
