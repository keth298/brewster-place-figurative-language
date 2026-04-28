# Brewster Place Figurative Language Analysis

Final paper project for **ENGL 1170: African American Short Stories**.

Analyzes Gloria Naylor's *The Women of Brewster Place* using two NLP pipelines to detect figurative language patterns and correlate them with character trait dimensions.

---

## Pipelines

### Pipeline 1 — Figurative Language Detection (`main.py`)
Detects similes and metaphors in each character's chapter using spaCy dependency parsing and the Brysbaert concreteness norms.

**Outputs:**
- `figurative_language_output.csv` — every detected instance with character, type, sentence, and position
- `summary_output.csv` — per-character counts and figurative language density (per 1000 words)

### Pipeline 2 — Character Trait Scoring & Correlation (`pipeline2.py`)
Scores each character on three trait dimensions and runs Pearson correlations against figurative language density.

**Traits measured:**
- **Racial consciousness** — keyword frequency (e.g. race, community, liberation)
- **Emotional register** — mean VADER sentiment score
- **Domestic/relational positioning** — keyword frequency (e.g. home, mother, marriage)

**Outputs:**
- `character_trait_scores.csv` — trait scores per character
- `merged_output.csv` — full merged dataset used for correlation analysis

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Place the novel text as `novel.txt` in the project root, then build the concreteness cache:

```bash
python setup_mrc.py
```

---

## Running

Pipeline 1 must be run before Pipeline 2:

```bash
python main.py
python pipeline2.py
```

---

## Key Finding

Emotional register has a strong positive correlation with figurative language density (r=0.88, p=0.02) — characters whose chapters carry more emotional sentiment use significantly more figurative language. Racial consciousness and domestic positioning show no significant relationship.
