import pytest
import spacy
from text_splitter import split_chapters, get_sentences_with_positions


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


SAMPLE_TEXT = """
Some preamble text here that belongs to no chapter.

The Block
This is the framing section and should be skipped entirely.

Mattie Michael
She walked down the street. The sun was bright. It was a good day.
The birds sang outside her window. She felt peaceful.

Etta Mae Johnson
She sang loud and clear. The music filled the room. Her voice rang out.
"""


def test_split_chapters_returns_expected_characters():
    chunks = split_chapters(SAMPLE_TEXT)
    assert "Mattie Michael" in chunks
    assert "Etta Mae Johnson" in chunks


def test_split_chapters_skips_the_block():
    chunks = split_chapters(SAMPLE_TEXT)
    assert "The Block" not in chunks


def test_split_chapters_content_belongs_to_correct_character():
    chunks = split_chapters(SAMPLE_TEXT)
    assert "walked" in chunks["Mattie Michael"]
    assert "sang" in chunks["Etta Mae Johnson"]
    assert "walked" not in chunks["Etta Mae Johnson"]


def test_split_chapters_case_insensitive_header():
    text = "\nMATTIE MICHAEL\nShe walked home.\n"
    chunks = split_chapters(text)
    assert "Mattie Michael" in chunks
    assert chunks["Mattie Michael"].strip() == "She walked home."


def test_split_chapters_warns_on_missing_header(capsys):
    text = "\nMattie Michael\nSome text here.\n"
    split_chapters(text)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "Etta Mae Johnson" in captured.out


def test_get_sentences_with_positions_three_thirds(nlp):
    # 9 sentences → 3 early, 3 middle, 3 late
    text = " ".join([f"Sentence number {i} ends here." for i in range(9)])
    result = get_sentences_with_positions(text, nlp)
    positions = [p for _, p in result]
    assert positions.count("early") >= 1
    assert positions.count("middle") >= 1
    assert positions.count("late") >= 1


def test_get_sentences_with_positions_single_sentence(nlp):
    result = get_sentences_with_positions("She walked home.", nlp)
    assert len(result) == 1
    assert result[0][1] == "early"


def test_get_sentences_returns_text_and_label(nlp):
    result = get_sentences_with_positions("She ran fast. He smiled.", nlp)
    for sent_text, label in result:
        assert isinstance(sent_text, str)
        assert label in ("early", "middle", "late")
