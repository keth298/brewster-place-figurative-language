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
        "thresholds": {"abstract_subject_max": 400},
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
