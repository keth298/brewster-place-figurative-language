import os
import spacy
import pandas as pd
from config import load_config
from text_splitter import split_chapters, get_sentences_with_positions
from trait_scorer import score_character
from correlation_analyzer import run_correlations, print_correlations


def main():
    cfg = load_config()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print("Config loaded.")

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000

    with open(cfg["text_file"], encoding="utf-8", errors="replace") as f:
        text = f.read()

    chapters = split_chapters(text)

    rows = []
    for character, chapter_text in chapters.items():
        sentences = get_sentences_with_positions(chapter_text, nlp)
        sentence_texts = [s[0] for s in sentences]
        scores = score_character(chapter_text, sentence_texts)
        rows.append({"character": character, **scores})

    trait_df = pd.DataFrame(rows)
    print(f"Scored {len(rows)} characters.")
    trait_df.to_csv(os.path.join(cfg["output_dir"], "character_trait_scores.csv"), index=False)

    summary_df = pd.read_csv(cfg["summary_file"])

    merged_df = trait_df.merge(summary_df, on="character", how="inner")
    if len(merged_df) < len(trait_df):
        missing = set(trait_df["character"]) - set(merged_df["character"])
        raise ValueError(
            f"Merge dropped {len(trait_df) - len(merged_df)} character(s); "
            f"check 'summary_file' for mismatched names: {missing}"
        )
    print("Running correlations...")
    merged_df.to_csv(os.path.join(cfg["output_dir"], "merged_output.csv"), index=False)

    results = run_correlations(merged_df)
    print_correlations(results)


if __name__ == "__main__":
    main()
