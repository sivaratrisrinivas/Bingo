import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))

import server

DOCS = [
    {
        "id": 1,
        "title": "Night of the Living Dead",
        "document": "Zombies besiege a farmhouse. " * 20,
        "score": 3.2,
    },
    {"id": 2, "title": "Dawn of the Dead", "document": "Survivors hide in a mall.", "score": 2.5},
]


def call(app, path, query="", method="GET"):
    status = {}

    def start_response(s, headers):
        status["code"] = int(s.split()[0])
        status["headers"] = dict(headers)

    environ = {"REQUEST_METHOD": method, "PATH_INFO": path, "QUERY_STRING": query, "wsgi.input": io.BytesIO()}
    body = b"".join(app(environ, start_response))
    return status["code"], status["headers"], body


def make(**extra):
    searchers = {"bm25": lambda q, k: DOCS[:k], **extra}
    return server.make_app(searchers, eval_summary={"modes": {}})


def test_index_page_has_csp():
    code, headers, body = call(make(), "/")
    assert code == 200
    assert b"one query, every retriever" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_config_lists_only_available_modes_in_order():
    app = make(semantic=lambda q, k: [], rrf=lambda q, k: [])
    code, _, body = call(app, "/api/config")
    assert code == 200
    assert [m["id"] for m in json.loads(body)["modes"]] == ["bm25", "semantic", "rrf"]


def test_search_returns_ranked_trimmed_items():
    code, _, body = call(make(), "/api/search", "q=zombies&k=2")
    data = json.loads(body)
    assert code == 200
    items = data["results"]["bm25"]["items"]
    assert [i["rank"] for i in items] == [1, 2]
    assert items[0]["title"] == "Night of the Living Dead"
    assert len(items[0]["snippet"]) <= server.SNIPPET_CHARS and items[0]["snippet"].endswith("…")
    assert data["results"]["bm25"]["ms"] >= 0


def test_cross_encoder_score_preferred():
    hits = [{"id": 9, "title": "X", "document": "d", "score": 0.1, "crossencoder_score": 7.5}]
    app = server.make_app({"rrf_cross_encoder": lambda q, k: hits}, eval_summary={})
    _, _, body = call(app, "/api/search", "q=x")
    assert json.loads(body)["results"]["rrf_cross_encoder"]["items"][0]["score"] == 7.5


def test_validation():
    app = make()
    assert call(app, "/api/search", "q=")[0] == 400
    assert call(app, "/api/search", "q=" + "a" * 201)[0] == 400
    assert call(app, "/api/search", "q=x&k=0")[0] == 400
    assert call(app, "/api/search", "q=x&k=abc")[0] == 400
    assert call(app, "/api/search", "q=x&modes=semantic")[0] == 400
    assert call(app, "/nope")[0] == 404
    assert call(app, "/api/search", "q=x", method="POST")[0] == 405


def test_one_failing_mode_does_not_break_the_others():
    def boom(q, k):
        raise RuntimeError("model missing")

    code, _, body = call(make(semantic=boom), "/api/search", "q=x")
    data = json.loads(body)
    assert code == 200
    assert data["results"]["semantic"]["error"] == "search failed"
    assert len(data["results"]["bm25"]["items"]) == 2


def test_eval_summary_reads_the_committed_results():
    summary = server.load_eval_summary()
    assert summary["n_queries"] == 834
    bm25 = summary["modes"]["bm25"]
    assert 0 < bm25["overall"]["hit@10"] <= 1
    assert "keyword_typo" in bm25["by_style"]


def test_missing_results_file(tmp_path):
    assert server.load_eval_summary(tmp_path / "nope.json") == {}
