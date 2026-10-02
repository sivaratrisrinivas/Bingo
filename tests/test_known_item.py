import random

from generate_queries import parse, sample_movies, split_for, title_leak, typo
from run_known_item import check, rank_of, summarize


def test_rank_of():
    res = [{"id": 5}, {"id": 9}]
    assert rank_of(9, res) == 2
    assert rank_of(1, res) is None


def test_summarize_hits_and_mrr():
    s = summarize([1, 3, None, 12])
    assert s["hit@1"] == 0.25
    assert s["hit@5"] == 0.5
    assert s["hit@10"] == 0.5
    assert s["mrr@10"] == round((1 + 1 / 3) / 4, 4)


def test_title_leak_ignores_common_words():
    assert title_leak("a bear in the woods", "The Bear") == ["bear"]
    assert title_leak("the story of a man", "The Man Who Knew") == ["man"]
    assert title_leak("space station race", "Bear Tales") == []


def test_typo_changes_text_but_keeps_word_count():
    rng = random.Random(1)
    q = "haunted lighthouse keeper storm"
    t = typo(q, rng)
    assert t != q and len(t.split()) == 4 and sorted(t) == sorted(q)


def test_parse_requires_all_styles():
    assert parse('{"keyword": "a", "descriptive": "b", "vague": "c"}') == {
        "keyword": "a", "descriptive": "b", "vague": "c"}
    assert parse('```json\n{"keyword": "a", "descriptive": "b"}\n```') is None
    assert parse("no json") is None


def test_split_is_stable_and_balanced():
    splits = [split_for(i) for i in range(1000)]
    assert splits == [split_for(i) for i in range(1000)]
    assert 400 < splits.count("test") < 600


def test_sample_is_stratified():
    movies = [{"id": i, "title": f"t{i}", "description": "x" * i} for i in range(90)]
    s = sample_movies(movies, 30, seed=7)
    assert len(s) == 30 and len({m["id"] for m in s}) == 30
    assert {m["desc_tier"] for m in s} == {"short", "medium", "long"}


def test_check_flags_regressions():
    report = {"configurations": {"bm25": {"by_split": {"test": {"hit@10": 0.5}}}}}
    assert check(report, {"bm25": {"test": {"hit@10": 0.6}}})
    assert not check(report, {"bm25": {"test": {"hit@10": 0.5}}})
