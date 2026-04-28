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
