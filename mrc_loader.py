import pickle


def load_mrc_cache(cache_path):
    """Load the pickled {word: concreteness} dict from cache_path."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def get_concreteness(word, cache):
    """Return the concreteness score for word (case-insensitive), or None if not found."""
    return cache.get(word.lower())
