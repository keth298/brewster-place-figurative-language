import pytest
import spacy
from simile_detector import detect


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


def test_detects_like_simile(nlp):
    chunks = {"Mattie Michael": [("Her voice was like honey.", "early")]}
    results = detect(chunks, nlp)
    assert len(results) == 1
    assert results[0]["type"] == "simile"
    assert results[0]["character"] == "Mattie Michael"
    assert results[0]["position"] == "early"
    assert "Her voice was like honey." in results[0]["sentence"]


def test_detects_as_as_simile(nlp):
    chunks = {"Cora Lee": [("She was as strong as an ox.", "middle")]}
    results = detect(chunks, nlp)
    assert len(results) >= 1


def test_filters_like_used_as_verb(nlp):
    chunks = {"Mattie Michael": [("She liked the music very much.", "early")]}
    results = detect(chunks, nlp)
    assert len(results) == 0


def test_filters_as_soon_as(nlp):
    chunks = {"Etta Mae Johnson": [("As soon as she arrived, the room fell silent.", "early")]}
    results = detect(chunks, nlp)
    assert len(results) == 0


def test_filters_as_well(nlp):
    chunks = {"Kiswana Browne": [("She danced as well.", "late")]}
    results = detect(chunks, nlp)
    assert len(results) == 0


def test_filters_as_if(nlp):
    chunks = {"Cora Lee": [("She walked as if she owned the place.", "early")]}
    results = detect(chunks, nlp)
    assert len(results) == 0


def test_filters_just_like_that(nlp):
    chunks = {"Lucielia Louise Turner": [("And just like that, it was over.", "middle")]}
    results = detect(chunks, nlp)
    assert len(results) == 0


def test_sentence_without_like_or_as_not_checked(nlp):
    chunks = {"The Two": [("She ran down the street.", "early")]}
    results = detect(chunks, nlp)
    assert len(results) == 0


def test_result_has_all_required_keys(nlp):
    chunks = {"Mattie Michael": [("Her tears fell like rain.", "late")]}
    results = detect(chunks, nlp)
    if results:
        assert set(results[0].keys()) == {"character", "sentence", "type", "position"}
