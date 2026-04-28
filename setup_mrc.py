import os
import pickle
import sys
from config import load_config


def parse_mrc(mrc_path):
    """Parse mrc2.dct into {word: concreteness_score}. Skips entries with CNC=0."""
    concreteness = {}
    with open(mrc_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            word = parts[1].lower().strip()
            if not word:
                continue
            fields = parts[0].split()
            if len(fields) < 10:
                continue
            try:
                cnc = int(fields[9])
            except ValueError:
                continue
            if cnc > 0:
                concreteness[word] = cnc
    return concreteness


def parse_brysbaert(path):
    """Parse Brysbaert et al. (2014) concreteness norms (tab-separated, Conc.M on 1-5 scale).

    Rescales to MRC 100-700 range: score = round(100 + (conc - 1) / 4 * 600).
    Skips words with unknown concreteness (Percent_known == 0).
    """
    concreteness = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        word_col = header.index("Word")
        conc_col = header.index("Conc.M")
        known_col = header.index("Percent_known")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(word_col, conc_col, known_col):
                continue
            word = parts[word_col].lower().strip()
            if not word:
                continue
            try:
                conc = float(parts[conc_col])
                known = float(parts[known_col])
            except ValueError:
                continue
            if known == 0:
                continue
            scaled = round(100 + (conc - 1) / 4 * 600)
            concreteness[word] = scaled
    return concreteness


def main():
    cfg = load_config()
    cache_path = cfg["mrc_cache"]

    brysbaert_path = "brysbaert_concreteness.txt"
    if os.path.exists(brysbaert_path):
        print(f"Parsing Brysbaert concreteness norms from {brysbaert_path} ...")
        concreteness = parse_brysbaert(brysbaert_path)
        print(f"Loaded {len(concreteness):,} words with concreteness ratings.")
    else:
        mrc_path = cfg["mrc_file"]
        if not os.path.exists(mrc_path):
            print(f"No concreteness data found.")
            print(f"  Option A: place brysbaert_concreteness.txt in this directory")
            print(f"  Option B: set 'mrc_file' in config.json to a valid mrc2.dct path")
            sys.exit(1)
        print(f"Parsing {mrc_path} ...")
        concreteness = parse_mrc(mrc_path)
        print(f"Loaded {len(concreteness):,} words with concreteness ratings.")

    with open(cache_path, "wb") as f:
        pickle.dump(concreteness, f)
    print(f"Saved cache to {cache_path}")


if __name__ == "__main__":
    main()
