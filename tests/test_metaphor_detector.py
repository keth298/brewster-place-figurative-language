import pytest
import spacy
from metaphor_detector import detect


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def mrc_cache():
    # "hope": abstract (200), "run": concrete (580)
    # "grief": abstract (180), "devour": concrete (520)
    # "dog": concrete (650), "bark": not-concrete-enough (480)
    return {
        "hope": 200,
        "run": 580,
        "grief": 180,
        "devour": 520,
        "dog": 650,
        "bark": 480,
    }


def test_detects_abstract_subject_concrete_verb(nlp, mrc_cache):
    # "hope" (200, abstract) subject of "runs" → lemma "run" (580, concrete)
    chunks = {"Mattie Michael": [("Hope runs through her veins.", "early")]}
    results = detect(chunks, nlp, mrc_cache, abstract_max=400, concrete_min=500)
    assert len(results) == 1
    assert results[0]["type"] == "metaphor"
    assert results[0]["character"] == "Mattie Michael"


def test_skips_concrete_subject(nlp, mrc_cache):
    # "dog" (650) is concrete — should not flag even if verb is concrete
    chunks = {"Cora Lee": [("The dog runs fast.", "early")]}
    results = detect(chunks, nlp, mrc_cache, abstract_max=400, concrete_min=500)
    assert len(results) == 0


def test_skips_abstract_verb_below_threshold(nlp, mrc_cache):
    # "dog" (650) concrete subject, "bark" (480) not concrete enough
    chunks = {"Etta Mae Johnson": [("The dog barks loudly.", "middle")]}
    results = detect(chunks, nlp, mrc_cache, abstract_max=400, concrete_min=500)
    assert len(results) == 0


def test_skips_missing_subject_in_cache(nlp, mrc_cache):
    # "sky" not in mrc_cache
    chunks = {"Kiswana Browne": [("The sky whispers secrets.", "late")]}
    results = detect(chunks, nlp, mrc_cache, abstract_max=400, concrete_min=500)
    assert len(results) == 0


def test_skips_missing_verb_in_cache(nlp, mrc_cache):
    # "hope" is in cache but "linger" is not
    chunks = {"The Two": [("Hope lingers in the air.", "early")]}
    results = detect(chunks, nlp, mrc_cache, abstract_max=400, concrete_min=500)
    assert len(results) == 0


def test_uses_lemma_fallback(nlp, mrc_cache):
    # "ran" is not in cache but "run" (580) is — lemma fallback should find it
    chunks = {"Lucielia Louise Turner": [("Hope ran through her veins.", "middle")]}
    results = detect(chunks, nlp, mrc_cache, abstract_max=400, concrete_min=500)
    assert len(results) == 1


def test_result_has_all_required_keys(nlp, mrc_cache):
    chunks = {"Mattie Michael": [("Grief devours her quietly.", "late")]}
    results = detect(chunks, nlp, mrc_cache, abstract_max=400, concrete_min=500)
    if results:
        assert set(results[0].keys()) == {"character", "sentence", "type", "position"}
        assert results[0]["position"] == "late"
