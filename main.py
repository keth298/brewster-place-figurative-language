import os
import spacy
import pandas as pd
from config import load_config
from mrc_loader import load_mrc_cache
from text_splitter import split_chapters, get_sentences_with_positions
from simile_detector import detect as detect_similes
from metaphor_detector import detect as detect_metaphors


def main():
    cfg = load_config()

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000  # handle long chapters

    print("Loading MRC cache...")
    mrc_cache = load_mrc_cache(cfg["mrc_cache"])
    print(f"  {len(mrc_cache):,} words loaded.")

    print("Reading novel text...")
    with open(cfg["text_file"], "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    print("Splitting chapters...")
    raw_chunks = split_chapters(text)
    print(f"  Found {len(raw_chunks)} chapters: {list(raw_chunks.keys())}")

    print("Segmenting sentences...")
    chunks = {
        character: get_sentences_with_positions(chapter_text, nlp)
        for character, chapter_text in raw_chunks.items()
    }

    thresholds = cfg["thresholds"]

    print("Detecting similes...")
    similes = detect_similes(chunks, nlp)
    print(f"  Found {len(similes)} similes.")

    print("Detecting metaphors...")
    metaphors = detect_metaphors(
        chunks,
        nlp,
        mrc_cache,
        abstract_max=thresholds["abstract_subject_max"],
        concrete_min=thresholds["concrete_verb_min"],
    )
    print(f"  Found {len(metaphors)} metaphors.")

    all_instances = similes + metaphors
    df = pd.DataFrame(all_instances, columns=["character", "sentence", "type", "position"])

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    instances_path = os.path.join(out_dir, "figurative_language_output.csv")
    df.to_csv(instances_path, index=False)
    print(f"\nWrote {len(df)} instances → {instances_path}")

    summary_rows = []
    for character, sentences in chunks.items():
        total_words = sum(len(s.split()) for s, _ in sentences)
        char_df = df[df["character"] == character]
        simile_count = int((char_df["type"] == "simile").sum())
        metaphor_count = int((char_df["type"] == "metaphor").sum())
        total = simile_count + metaphor_count
        freq = round(total / total_words * 1000, 2) if total_words > 0 else 0.0
        summary_rows.append({
            "character": character,
            "simile_count": simile_count,
            "metaphor_count": metaphor_count,
            "total_words": total_words,
            "figurative_per_1000_words": freq,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary_output.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSummary:\n{summary_df.to_string(index=False)}")
    print(f"\nWrote summary → {summary_path}")


if __name__ == "__main__":
    main()
