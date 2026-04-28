def detect(chunks, nlp, mrc_cache, abstract_max=400, concrete_min=500):
    """Detect metaphors in chunks using MRC concreteness scores.

    chunks: {character: [(sentence_text, position_label), ...]}
    mrc_cache: {word: concreteness_score} dict from mrc_loader.load_mrc_cache()
    Returns list of dicts with keys: character, sentence, type, position.
    """
    results = []
    for character, sentences in chunks.items():
        for sent_text, position in sentences:
            doc = nlp(sent_text)
            for token in doc:
                if token.dep_ != "nsubj":
                    continue
                head = token.head
                if head.pos_ != "VERB":
                    continue
                subj_cnc = _lookup(token, mrc_cache)
                if subj_cnc is None:
                    continue
                verb_cnc = _lookup(head, mrc_cache)
                if verb_cnc is None:
                    continue
                if subj_cnc < abstract_max and verb_cnc > concrete_min:
                    results.append({
                        "character": character,
                        "sentence": sent_text,
                        "type": "metaphor",
                        "position": position,
                    })
                    break  # at most one metaphor entry per sentence
    return results


def _lookup(token, cache):
    """Try raw word text then lemma against cache. Returns int or None."""
    word = token.text.lower()
    if word in cache:
        return cache[word]
    lemma = token.lemma_.lower()
    if lemma in cache:
        return cache[lemma]
    return None
