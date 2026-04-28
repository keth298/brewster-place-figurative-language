# Character Trait Scoring — Design Spec
**Date:** 2026-04-27
**Project:** FWS Final Project — Pipeline 2
**Novel:** Gloria Naylor's *The Women of Brewster Place*

---

## Overview

A modular Python pipeline that scores each chapter character on three trait dimensions (racial consciousness, emotional register, domestic/relational positioning), then merges those scores with Pipeline 1's figurative language frequency output and runs Pearson correlations between each trait and figurative language density.

---

## File Structure

**New files:**
```
FWS_Final_Project/
├── trait_scorer.py              # keyword scoring + VADER sentiment
├── correlation_analyzer.py      # CSV merge + Pearson correlations
├── pipeline2.py                 # entry point
├── tests/test_trait_scorer.py
└── tests/test_correlation_analyzer.py
```

**Reused from Pipeline 1 (text_splitter.py unchanged):**
- `config.py` — `load_config()` — **modified**: `"summary_file"` added to `required_keys` list
- `text_splitter.py` — `split_chapters()`, `get_sentences_with_positions()` — no changes
- `config.json` — extended with two new keys (see below)

---

## Configuration

`config.json` gains two new keys:

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

`summary_file` points to Pipeline 1's `summary_output.csv`, which contains the `figurative_per_1000_words` column used in the correlation step.

`config.py`'s `load_config()` must be updated to validate the presence of `summary_file`.

**`requirements.txt` additions:**
```
nltk>=3.7.0
scipy>=1.9.0
```

---

## Trait Scorer (`trait_scorer.py`)

### Keyword Lists

```python
RACIAL_KEYWORDS = {
    "black", "african", "afro", "revolution", "community", "movement",
    "white", "people", "race", "struggle", "protest", "liberation",
    "heritage", "identity"
}

DOMESTIC_KEYWORDS = {
    "husband", "baby", "child", "children", "home", "clean", "cook",
    "dinner", "mother", "marriage", "pregnant", "eugene", "basil"
}
```

All matching is case-insensitive with word-boundary enforcement.

### Functions

**`score_keywords(text, keywords) -> float`**
- Compiles a single combined regex: `\b(?:word1|word2|...)\b` with `re.IGNORECASE`
- Counts all matches in `text`
- Returns `count / len(text.split()) * 1000` (frequency per 1000 words)
- Returns `0.0` if `text` is empty

**`score_sentiment(sentences) -> float`**
- Initialises `SentimentIntensityAnalyzer` from `nltk.sentiment.vader`
- Runs analyser on each sentence, collects `compound` scores
- Returns the mean compound score across all sentences
- Returns `0.0` if `sentences` is empty

**`score_character(chapter_text, sentences) -> dict`**
- Calls `score_keywords(chapter_text, RACIAL_KEYWORDS)` → `racial_consciousness`
- Calls `score_keywords(chapter_text, DOMESTIC_KEYWORDS)` → `domestic_score`
- Calls `score_sentiment(sentences)` → `emotional_register`
- Returns `{"racial_consciousness": float, "emotional_register": float, "domestic_score": float}`

### Word Count Convention
`len(text.split())` — consistent with Pipeline 1's summary table.

---

## Correlation Analyzer (`correlation_analyzer.py`)

**`run_correlations(merged_df) -> list[dict]`**
- Receives the already-merged DataFrame (trait scores + Pipeline 1 summary columns)
- Runs `scipy.stats.pearsonr` for each of the three trait columns against `figurative_per_1000_words`
- Returns: `[{"trait": str, "r": float, "p_value": float}, ...]`

**`print_correlations(results)`**
- Prints formatted output to console:
```
Correlation Results:
  racial_consciousness  vs figurative_per_1000_words:  r=0.xx, p=0.xx
  emotional_register    vs figurative_per_1000_words:  r=0.xx, p=0.xx
  domestic_score        vs figurative_per_1000_words:  r=0.xx, p=0.xx
```

---

## Pipeline Entry Point (`pipeline2.py`)

Execution order:

1. `nltk.download("vader_lexicon", quiet=True)` — silent download on first run
2. `load_config()` — validate config including `summary_file`
3. `spacy.load("en_core_web_sm")`, set `nlp.max_length = 2_000_000`
4. Read novel text file (utf-8, errors="replace")
5. `split_chapters(text)` → per-character chapter text
6. For each character: `get_sentences_with_positions(chapter_text, nlp)` → extract sentence texts → `score_character(chapter_text, sentences)`
7. Build `character_trait_scores.csv` in `output_dir`
8. Load `summary_output.csv` from `cfg["summary_file"]`
9. Merge on `character` column (inner join — both CSVs have exactly the 6 chapter characters)
10. Save merged DataFrame as `merged_output.csv` in `output_dir`
11. `run_correlations(merged_df)` → `print_correlations(results)`

---

## Output Files

### `character_trait_scores.csv`
One row per character.

| column | description |
|---|---|
| `character` | chapter section name |
| `racial_consciousness` | keyword freq per 1000 words |
| `emotional_register` | mean VADER compound score (-1 to 1) |
| `domestic_score` | keyword freq per 1000 words |

### `merged_output.csv`
All columns from both CSVs joined on `character`:
`character`, `racial_consciousness`, `emotional_register`, `domestic_score`, `simile_count`, `metaphor_count`, `total_words`, `figurative_per_1000_words`

---

## Dependencies

- `nltk` + `vader_lexicon` (`nltk.download("vader_lexicon", quiet=True)`)
- `scipy`
- `pandas`
- `re` (stdlib)
- `spacy` + `en_core_web_sm` (already installed)

---

## Run Order

```bash
# Prerequisite: Pipeline 1 must have been run first to produce summary_output.csv
python pipeline2.py
```
