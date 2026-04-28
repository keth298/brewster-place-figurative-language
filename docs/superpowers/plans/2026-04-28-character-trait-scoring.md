# Character Trait Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Pipeline 2 — a modular Python pipeline that scores each chapter character on three trait dimensions (racial consciousness, VADER sentiment, domestic positioning), merges results with Pipeline 1's figurative language frequency output, and runs Pearson correlations between each trait and figurative language density.

**Architecture:** Three new modules follow the same pattern as Pipeline 1: `trait_scorer.py` handles keyword frequency and VADER sentiment, `correlation_analyzer.py` merges CSVs and runs correlations, and `pipeline2.py` orchestrates everything. `config.py` and `text_splitter.py` are reused directly; `config.py` gets one new required key (`summary_file`).

**Tech Stack:** Python 3.8+, spaCy (`en_core_web_sm`), nltk (VADER), pandas, scipy.stats, re (stdlib), pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add nltk, scipy |
| `config.json` | Modify | Add `summary_file` key |
| `config.py` | Modify | Add `"summary_file"` to required_keys |
| `tests/test_config.py` | Modify | Add `summary_file` to all full-config fixtures + new test |
| `trait_scorer.py` | Create | Keyword scoring + VADER sentiment |
| `correlation_analyzer.py` | Create | CSV merge + Pearson correlations + console output |
| `pipeline2.py` | Create | Entry point — orchestrates all modules |
| `tests/test_trait_scorer.py` | Create | Tests for all scoring functions |
| `tests/test_correlation_analyzer.py` | Create | Tests for merge + correlation logic |

---

## Task 1: Update Dependencies and Config

**Files:**
- Modify: `requirements.txt`
- Modify: `config.json`
- Modify: `config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add new deps to `requirements.txt`**

The file currently ends after `pytest>=7.0.0`. Append two lines so the full file reads:

```
spacy>=3.0.0
pandas>=1.3.0
pytest>=7.0.0
nltk>=3.7.0
scipy>=1.9.0
```

- [ ] **Step 2: Install new dependencies**

Run:
```bash
cd /Users/kimet/FWS_Final_Project && pip install nltk scipy
```

Expected: `Successfully installed` messages (or `already satisfied` if already present).

- [ ] **Step 3: Update `tests/test_config.py`**

Four existing tests create full-config dicts that don't include `summary_file` — they will break once `config.py` requires it. Replace the entire file:

```python
import json
import pytest
from config import load_config


def test_load_config_valid(tmp_path):
    novel = tmp_path / "novel.txt"
    novel.write_text("text")
    cfg_data = {
        "text_file": str(novel),
        "mrc_file": "mrc2.dct",
        "mrc_cache": "mrc.pkl",
        "output_dir": ".",
        "summary_file": "summary_output.csv",
        "thresholds": {"abstract_subject_max": 400, "concrete_verb_min": 500},
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg_data))
    cfg = load_config(str(cfg_file))
    assert cfg["thresholds"]["abstract_subject_max"] == 400
    assert cfg["thresholds"]["concrete_verb_min"] == 500
    assert cfg["summary_file"] == "summary_output.csv"


def test_load_config_missing_required_key(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"text_file": "x"}))
    with pytest.raises(KeyError, match="mrc_file"):
        load_config(str(cfg_file))


def test_load_config_missing_summary_file_key(tmp_path):
    novel = tmp_path / "novel.txt"
    novel.write_text("text")
    cfg_data = {
        "text_file": str(novel),
        "mrc_file": "mrc2.dct",
        "mrc_cache": "mrc.pkl",
        "output_dir": ".",
        # summary_file intentionally omitted
        "thresholds": {"abstract_subject_max": 400, "concrete_verb_min": 500},
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg_data))
    with pytest.raises(KeyError, match="summary_file"):
        load_config(str(cfg_file))


def test_load_config_missing_threshold_key(tmp_path):
    novel = tmp_path / "novel.txt"
    novel.write_text("text")
    cfg_data = {
        "text_file": str(novel),
        "mrc_file": "mrc2.dct",
        "mrc_cache": "mrc.pkl",
        "output_dir": ".",
        "summary_file": "summary_output.csv",
        "thresholds": {"abstract_subject_max": 400},  # missing concrete_verb_min
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg_data))
    with pytest.raises(KeyError, match="concrete_verb_min"):
        load_config(str(cfg_file))


def test_load_config_text_file_not_found(tmp_path):
    cfg_data = {
        "text_file": "/nonexistent/novel.txt",
        "mrc_file": "mrc2.dct",
        "mrc_cache": "mrc.pkl",
        "output_dir": ".",
        "summary_file": "summary_output.csv",
        "thresholds": {"abstract_subject_max": 400, "concrete_verb_min": 500},
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg_data))
    with pytest.raises(FileNotFoundError, match="text_file"):
        load_config(str(cfg_file))


def test_load_config_thresholds_not_dict(tmp_path):
    novel = tmp_path / "novel.txt"
    novel.write_text("text")
    cfg_data = {
        "text_file": str(novel),
        "mrc_file": "mrc2.dct",
        "mrc_cache": "mrc.pkl",
        "output_dir": ".",
        "summary_file": "summary_output.csv",
        "thresholds": None,
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg_data))
    with pytest.raises(TypeError, match="thresholds"):
        load_config(str(cfg_file))
```

- [ ] **Step 4: Run tests to verify they fail**

Run:
```bash
cd /Users/kimet/FWS_Final_Project && python -m pytest tests/test_config.py -v
```

Expected: `test_load_config_missing_summary_file_key` FAILS (KeyError not yet raised by config.py). All other tests PASS — `test_load_config_valid` passes because `summary_file` is present in the dict even before it's a required key.

- [ ] **Step 5: Update `config.py`**

Add `"summary_file"` to `required_keys` (line 9):

```python
import json
import os


def load_config(path="config.json"):
    with open(path) as f:
        cfg = json.load(f)

    required_keys = ["text_file", "mrc_file", "mrc_cache", "output_dir", "summary_file", "thresholds"]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(key)

    if not isinstance(cfg["thresholds"], dict):
        raise TypeError(f"'thresholds' must be a dict, got {type(cfg['thresholds']).__name__}")

    required_thresholds = ["abstract_subject_max", "concrete_verb_min"]
    for key in required_thresholds:
        if key not in cfg["thresholds"]:
            raise KeyError(key)

    if not os.path.exists(cfg["text_file"]):
        raise FileNotFoundError(f"text_file not found: {cfg['text_file']}")

    return cfg
```

- [ ] **Step 6: Run config tests to verify all pass**

Run:
```bash
cd /Users/kimet/FWS_Final_Project && python -m pytest tests/test_config.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 7: Update `config.json`**

```json
{
  "text_file": "path/to/novel.txt",
  "mrc_file": "path/to/mrc2.dct",
  "mrc_cache": "mrc_concreteness.pkl",
  "output_dir": ".",
  "summary_file": "summary_output.csv",
  "thresholds": {
    "abstract_subject_max": 400,
    "concrete_verb_min": 500
  }
}
```

- [ ] **Step 8: Run full test suite to confirm no regressions**

Run:
```bash
cd /Users/kimet/FWS_Final_Project && python -m pytest -v
```

Expected: all 36 tests PASS (one new config test = 37 total).

- [ ] **Step 9: Commit**

```bash
cd /Users/kimet/FWS_Final_Project && git add requirements.txt config.json config.py tests/test_config.py && git commit -m "feat: add summary_file config key and Pipeline 2 dependencies"
```

---

## Task 2: Trait Scorer

**Files:**
- Create: `trait_scorer.py`
- Create: `tests/test_trait_scorer.py`

- [ ] **Step 1: Write `tests/test_trait_scorer.py`**

```python
import pytest
from trait_scorer import (
    score_keywords,
    score_sentiment,
    score_character,
    RACIAL_KEYWORDS,
    DOMESTIC_KEYWORDS,
)


def test_score_keywords_counts_correctly():
    # "black", "community", "struggle", "liberation" = 4 matches in 11 words
    text = "The black community stood together in the struggle for liberation."
    result = score_keywords(text, RACIAL_KEYWORDS)
    assert result == pytest.approx(4 / 11 * 1000, rel=1e-3)


def test_score_keywords_case_insensitive():
    text = "BLACK people and African heritage matter."
    result = score_keywords(text, RACIAL_KEYWORDS)
    # "BLACK", "people", "African", "heritage" = 4 matches
    assert result > 0


def test_score_keywords_word_boundary():
    # "blackbird" should NOT match "black"; only "community" matches
    text = "The blackbird flew over the community."
    result = score_keywords(text, RACIAL_KEYWORDS)
    assert result == pytest.approx(1 / 7 * 1000, rel=1e-3)


def test_score_keywords_empty_text():
    assert score_keywords("", RACIAL_KEYWORDS) == 0.0


def test_score_keywords_no_matches():
    assert score_keywords("She walked home quietly.", RACIAL_KEYWORDS) == 0.0


def test_score_sentiment_positive():
    sentences = ["She felt wonderful and joyful today."]
    result = score_sentiment(sentences)
    assert result > 0.0


def test_score_sentiment_negative():
    sentences = ["Everything was terrible and awful and painful."]
    result = score_sentiment(sentences)
    assert result < 0.0


def test_score_sentiment_empty_list():
    assert score_sentiment([]) == 0.0


def test_score_sentiment_returns_float_in_range():
    sentences = ["She walked home slowly.", "The sun was setting over the hills."]
    result = score_sentiment(sentences)
    assert -1.0 <= result <= 1.0


def test_score_character_returns_all_keys():
    result = score_character("She fought for black liberation.", ["She fought for black liberation."])
    assert set(result.keys()) == {"racial_consciousness", "emotional_register", "domestic_score"}


def test_score_character_racial_consciousness():
    text = "The black community fought for liberation and race identity."
    result = score_character(text, [text])
    assert result["racial_consciousness"] > 0


def test_score_character_domestic_score():
    text = "She cleaned the home and cooked dinner for her husband and children."
    result = score_character(text, [text])
    assert result["domestic_score"] > 0


def test_score_character_emotional_register_range():
    text = "She walked quietly through the evening light."
    result = score_character(text, [text])
    assert -1.0 <= result["emotional_register"] <= 1.0


def test_domestic_keywords_contains_expected():
    assert "husband" in DOMESTIC_KEYWORDS
    assert "children" in DOMESTIC_KEYWORDS
    assert "eugene" in DOMESTIC_KEYWORDS


def test_racial_keywords_contains_expected():
    assert "black" in RACIAL_KEYWORDS
    assert "liberation" in RACIAL_KEYWORDS
    assert "community" in RACIAL_KEYWORDS
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/kimet/FWS_Final_Project && python -m pytest tests/test_trait_scorer.py -v
```

Expected: `ModuleNotFoundError: No module named 'trait_scorer'`

- [ ] **Step 3: Write `trait_scorer.py`**

```python
import re
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

RACIAL_KEYWORDS = {
    "black", "african", "afro", "revolution", "community", "movement",
    "white", "people", "race", "struggle", "protest", "liberation",
    "heritage", "identity",
}

DOMESTIC_KEYWORDS = {
    "husband", "baby", "child", "children", "home", "clean", "cook",
    "dinner", "mother", "marriage", "pregnant", "eugene", "basil",
}

_sia = SentimentIntensityAnalyzer()


def score_keywords(text, keywords):
    """Return keyword frequency per 1000 words (case-insensitive, word-boundary matched)."""
    if not text or not text.split():
        return 0.0
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b",
        re.IGNORECASE,
    )
    count = len(pattern.findall(text))
    return count / len(text.split()) * 1000


def score_sentiment(sentences):
    """Return mean VADER compound score across all sentences. Returns 0.0 if empty."""
    if not sentences:
        return 0.0
    scores = [_sia.polarity_scores(s)["compound"] for s in sentences]
    return sum(scores) / len(scores)


def score_character(chapter_text, sentences):
    """Return trait scores dict for a single character's chapter text."""
    return {
        "racial_consciousness": score_keywords(chapter_text, RACIAL_KEYWORDS),
        "emotional_register": score_sentiment(sentences),
        "domestic_score": score_keywords(chapter_text, DOMESTIC_KEYWORDS),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/kimet/FWS_Final_Project && python -m pytest tests/test_trait_scorer.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/kimet/FWS_Final_Project && git add trait_scorer.py tests/test_trait_scorer.py && git commit -m "feat: trait scorer with keyword and VADER sentiment scoring"
```

---

## Task 3: Correlation Analyzer

**Files:**
- Create: `correlation_analyzer.py`
- Create: `tests/test_correlation_analyzer.py`

- [ ] **Step 1: Write `tests/test_correlation_analyzer.py`**

```python
import pandas as pd
import pytest
from correlation_analyzer import run_correlations, print_correlations


@pytest.fixture
def perfect_df():
    # racial_consciousness perfectly correlates with figurative_per_1000_words
    # emotional_register perfectly negatively correlates
    return pd.DataFrame({
        "character": ["A", "B", "C", "D", "E", "F"],
        "racial_consciousness": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "emotional_register": [0.6, 0.4, 0.2, -0.2, -0.4, -0.6],
        "domestic_score": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        "figurative_per_1000_words": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


def test_run_correlations_returns_three_results(perfect_df):
    results = run_correlations(perfect_df)
    assert len(results) == 3


def test_run_correlations_has_required_keys(perfect_df):
    results = run_correlations(perfect_df)
    for res in results:
        assert set(res.keys()) == {"trait", "r", "p_value"}


def test_run_correlations_trait_names(perfect_df):
    results = run_correlations(perfect_df)
    traits = [r["trait"] for r in results]
    assert "racial_consciousness" in traits
    assert "emotional_register" in traits
    assert "domestic_score" in traits


def test_run_correlations_perfect_positive(perfect_df):
    results = run_correlations(perfect_df)
    racial = next(r for r in results if r["trait"] == "racial_consciousness")
    assert racial["r"] == pytest.approx(1.0, abs=1e-6)
    assert racial["p_value"] < 0.05


def test_run_correlations_perfect_negative(perfect_df):
    results = run_correlations(perfect_df)
    sentiment = next(r for r in results if r["trait"] == "emotional_register")
    assert sentiment["r"] == pytest.approx(-1.0, abs=1e-4)


def test_run_correlations_r_in_range(perfect_df):
    results = run_correlations(perfect_df)
    for res in results:
        assert -1.0 <= res["r"] <= 1.0


def test_print_correlations_outputs_all_traits(perfect_df, capsys):
    results = run_correlations(perfect_df)
    print_correlations(results)
    captured = capsys.readouterr()
    assert "racial_consciousness" in captured.out
    assert "emotional_register" in captured.out
    assert "domestic_score" in captured.out
    assert "figurative_per_1000_words" in captured.out


def test_print_correlations_shows_r_and_p(perfect_df, capsys):
    results = run_correlations(perfect_df)
    print_correlations(results)
    captured = capsys.readouterr()
    assert "r=" in captured.out
    assert "p=" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/kimet/FWS_Final_Project && python -m pytest tests/test_correlation_analyzer.py -v
```

Expected: `ModuleNotFoundError: No module named 'correlation_analyzer'`

- [ ] **Step 3: Write `correlation_analyzer.py`**

```python
import pandas as pd
from scipy import stats


def run_correlations(merged_df):
    """Run Pearson correlations between each trait column and figurative_per_1000_words.

    Returns list of dicts: [{"trait": str, "r": float, "p_value": float}, ...]
    """
    trait_cols = ["racial_consciousness", "emotional_register", "domestic_score"]
    results = []
    for trait in trait_cols:
        r, p = stats.pearsonr(merged_df[trait], merged_df["figurative_per_1000_words"])
        results.append({"trait": trait, "r": float(r), "p_value": float(p)})
    return results


def print_correlations(results):
    """Print formatted correlation results to console."""
    print("\nCorrelation Results:")
    for res in results:
        print(
            f"  {res['trait']:<25} vs figurative_per_1000_words:"
            f"  r={res['r']:.4f}, p={res['p_value']:.4f}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/kimet/FWS_Final_Project && python -m pytest tests/test_correlation_analyzer.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/kimet/FWS_Final_Project && git add correlation_analyzer.py tests/test_correlation_analyzer.py && git commit -m "feat: correlation analyzer with Pearson correlation and formatted output"
```

---

## Task 4: Pipeline 2 Entry Point

**Files:**
- Create: `pipeline2.py`

- [ ] **Step 1: Write `pipeline2.py`**

```python
import os
import spacy
import pandas as pd
from config import load_config
from text_splitter import split_chapters, get_sentences_with_positions
from trait_scorer import score_character
from correlation_analyzer import run_correlations, print_correlations


def main():
    cfg = load_config()

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000

    print("Reading novel text...")
    with open(cfg["text_file"], "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    print("Splitting chapters...")
    raw_chunks = split_chapters(text)
    print(f"  Found {len(raw_chunks)} chapters: {list(raw_chunks.keys())}")

    print("Scoring characters...")
    rows = []
    for character, chapter_text in raw_chunks.items():
        sentences = [s for s, _ in get_sentences_with_positions(chapter_text, nlp)]
        scores = score_character(chapter_text, sentences)
        scores["character"] = character
        rows.append(scores)

    trait_df = pd.DataFrame(
        rows, columns=["character", "racial_consciousness", "emotional_register", "domestic_score"]
    )

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    trait_path = os.path.join(out_dir, "character_trait_scores.csv")
    trait_df.to_csv(trait_path, index=False)
    print(f"Wrote trait scores → {trait_path}")

    print("Loading Pipeline 1 summary...")
    summary_df = pd.read_csv(cfg["summary_file"])

    merged_df = trait_df.merge(summary_df, on="character", how="inner")

    merged_path = os.path.join(out_dir, "merged_output.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"Wrote merged output → {merged_path}")

    results = run_correlations(merged_df)
    print_correlations(results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite to confirm no regressions**

Run:
```bash
cd /Users/kimet/FWS_Final_Project && python -m pytest -v
```

Expected: all tests PASS (37 from Pipeline 1 + 14 trait scorer + 8 correlation = 59 total).

- [ ] **Step 3: Run a smoke test**

Update `config.json` to point `text_file` at `test_novel.txt` and `summary_file` at `summary_output.csv`:

```json
{
  "text_file": "test_novel.txt",
  "mrc_file": "path/to/mrc2.dct",
  "mrc_cache": "mrc_concreteness.pkl",
  "output_dir": ".",
  "summary_file": "summary_output.csv",
  "thresholds": {
    "abstract_subject_max": 400,
    "concrete_verb_min": 500
  }
}
```

Run:
```bash
cd /Users/kimet/FWS_Final_Project && python pipeline2.py
```

Expected: no errors; `character_trait_scores.csv` and `merged_output.csv` written; correlation results printed to console showing r and p values for all three trait pairs.

Verify outputs exist and have content:
```bash
cd /Users/kimet/FWS_Final_Project && cat character_trait_scores.csv && echo "---" && head -3 merged_output.csv
```

Expected: `character_trait_scores.csv` has 6 data rows (one per chapter character); `merged_output.csv` has 6 rows with all trait + summary columns.

- [ ] **Step 4: Restore `config.json` to placeholder paths**

```json
{
  "text_file": "path/to/novel.txt",
  "mrc_file": "path/to/mrc2.dct",
  "mrc_cache": "mrc_concreteness.pkl",
  "output_dir": ".",
  "summary_file": "summary_output.csv",
  "thresholds": {
    "abstract_subject_max": 400,
    "concrete_verb_min": 500
  }
}
```

- [ ] **Step 5: Commit**

```bash
cd /Users/kimet/FWS_Final_Project && git add pipeline2.py config.json && git commit -m "feat: pipeline 2 entry point for character trait scoring and correlation"
```

---

## Run Order (Final)

```bash
# Prerequisite: Pipeline 1 must have been run first to produce summary_output.csv
python pipeline2.py

# Outputs:
#   character_trait_scores.csv  — one row per character, three trait scores
#   merged_output.csv           — trait scores + Pipeline 1 summary merged on character
#   (console)                   — Pearson r and p-values for all three trait correlations
```
