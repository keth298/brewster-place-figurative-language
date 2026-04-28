import pickle
import pytest
from mrc_loader import load_mrc_cache, get_concreteness


def test_load_and_lookup(tmp_path):
    cache = {"able": 413, "run": 580, "grief": 180}
    cache_file = tmp_path / "mrc.pkl"
    cache_file.write_bytes(pickle.dumps(cache))
    loaded = load_mrc_cache(str(cache_file))
    assert get_concreteness("able", loaded) == 413
    assert get_concreteness("run", loaded) == 580


def test_lookup_is_case_insensitive(tmp_path):
    cache = {"able": 413}
    cache_file = tmp_path / "mrc.pkl"
    cache_file.write_bytes(pickle.dumps(cache))
    loaded = load_mrc_cache(str(cache_file))
    assert get_concreteness("ABLE", loaded) == 413
    assert get_concreteness("Able", loaded) == 413


def test_lookup_missing_word_returns_none(tmp_path):
    cache = {"able": 413}
    cache_file = tmp_path / "mrc.pkl"
    cache_file.write_bytes(pickle.dumps(cache))
    loaded = load_mrc_cache(str(cache_file))
    assert get_concreteness("nonexistent", loaded) is None


def test_load_returns_dict(tmp_path):
    cache = {}
    cache_file = tmp_path / "mrc.pkl"
    cache_file.write_bytes(pickle.dumps(cache))
    loaded = load_mrc_cache(str(cache_file))
    assert isinstance(loaded, dict)
