import math
import random

import pytest
from lib import keyword_search
from lib.hybrid_search import normalize_scores, rrf_combine_search_results, rrf_score
from lib.keyword_search import BM25_B, BM25_K1, InvertedIndex, tokenize_text

MOVIES = [
    {"id": 1, "title": "Bear Tales", "description": "A brown bear steals honey from a village."},
    {"id": 2, "title": "Space Run", "description": "Astronauts race to fix a broken station."},
    {"id": 3, "title": "Zombie Town", "description": "A zombie apocalypse hits a quiet town."},
    {"id": 4, "title": "Honey Heist", "description": "Thieves plan to steal honey, then a bear shows up."},
    {"id": 5, "title": "Quiet Night", "description": "Nothing happens at all."},
]


@pytest.fixture
def index(monkeypatch):
    monkeypatch.setattr(keyword_search, "load_movies", lambda: MOVIES)
    idx = InvertedIndex()
    idx.build()
    return idx


def brute_force(idx, query):
    """Score every document with textbook BM25 (single stemming)."""
    toks = tokenize_text(query)
    n = len(idx.docmap)
    avg = sum(idx.doc_lengths.values()) / n
    scores = []
    for doc_id in idx.docmap:
        s = 0.0
        for t in toks:
            df = len(idx.index.get(t, ()))
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            tf = idx.term_frequencies[doc_id][t]
            norm = 1 - BM25_B + BM25_B * idx.doc_lengths[doc_id] / avg
            s += tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm) * idf
        scores.append((doc_id, round(s, 3)))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def test_tokenize_lowercases_drops_stopwords_and_stems():
    assert tokenize_text("The Bears are RUNNING!") == ["bear", "run"]


def test_tokenize_empty():
    assert tokenize_text("") == []


@pytest.mark.parametrize("query", ["bear honey", "zombie apocalypse", "quiet", "nothing here xyz", ""])
def test_bm25_matches_brute_force_full_ranking(index, query):
    got = [(r["id"], r["score"]) for r in index.bm25_search(query, 10)]
    assert got == brute_force(index, query)


def test_bm25_matches_brute_force_random_queries(index):
    rng = random.Random(0)
    vocab = " ".join(m["title"] + " " + m["description"] for m in MOVIES).split()
    for _ in range(50):
        q = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 5)))
        assert [(r["id"], r["score"]) for r in index.bm25_search(q, 10)] == brute_force(index, q)


def test_bm25_finds_stemmed_terms(index):
    # "apocalypse" stems to "apocalyps"; the old code stemmed twice and lost it.
    assert index.bm25_search("apocalypse", 1)[0]["id"] == 3


def test_bm25_limit_and_validation(index):
    assert len(index.bm25_search("bear", 2)) == 2
    assert index.bm25_search("bear", 0) == []
    with pytest.raises(ValueError):
        index.bm25_search("bear", -1)
    with pytest.raises(TypeError):
        index.bm25_search(None, 5)


def test_normalize_scores():
    assert normalize_scores([]) == []
    assert normalize_scores([2.0, 2.0]) == [1.0, 1.0]
    assert normalize_scores([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_rrf_combines_ranks():
    bm25 = [{"id": 1, "title": "a"}, {"id": 2, "title": "b"}]
    sem = [{"id": 2, "title": "b"}, {"id": 3, "title": "c"}]
    fused = rrf_combine_search_results(bm25, sem, k=60)
    assert fused[0]["id"] == 2
    assert fused[0]["score"] == pytest.approx(rrf_score(2) + rrf_score(1), abs=1e-3)  # scores are rounded to 3 places
    assert {r["id"] for r in fused} == {1, 2, 3}
