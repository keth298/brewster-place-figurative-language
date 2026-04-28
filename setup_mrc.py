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


def main():
    cfg = load_config()
    mrc_path = cfg["mrc_file"]
    cache_path = cfg["mrc_cache"]

    if not os.path.exists(mrc_path):
        print(f"MRC file not found at: {mrc_path}")
        print()
        print("Download mrc2.dct from the MRC Psycholinguistic Database:")
        print("  Search: 'MRC Psycholinguistic Database mrc2.dct download'")
        print("  UWA page: http://websites.psychology.uwa.edu.au/school/MRCDatabase/")
        print("  Alternative: search GitHub for 'mrc2.dct' (several NLP repos mirror it)")
        print()
        print(f"Then set 'mrc_file' in config.json to the path of the downloaded file.")
        sys.exit(1)

    print(f"Parsing {mrc_path} ...")
    concreteness = parse_mrc(mrc_path)
    print(f"Loaded {len(concreteness):,} words with concreteness ratings.")

    with open(cache_path, "wb") as f:
        pickle.dump(concreteness, f)
    print(f"Saved cache to {cache_path}")


if __name__ == "__main__":
    main()
