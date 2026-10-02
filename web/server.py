"""Side-by-side search demo: one query, every retrieval mode, plus the eval numbers.

    uv run python web/server.py                 # all four modes (builds embeddings on first run)
    uv run python web/server.py --modes bm25    # keyword only, starts in about a second

Standard library only (wsgiref), so it adds no dependencies. The app is a plain
WSGI callable built by make_app(), which is what the tests drive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = Path(__file__).with_name("index.html")
RESULTS_JSON = ROOT / "eval" / "known_item" / "results.json"

MODES = ("bm25", "semantic", "rrf", "rrf_cross_encoder")
LABELS = {
    "bm25": "BM25",
    "semantic": "Semantic",
    "rrf": "Hybrid (RRF)",
    "rrf_cross_encoder": "RRF + cross-encoder",
}
MAX_QUERY_CHARS = 200
MAX_K = 20
SNIPPET_CHARS = 180

Searcher = Callable[[str, int], list[dict]]


def load_eval_summary(path: Path = RESULTS_JSON) -> dict:
    """The headline numbers from the known-item eval, trimmed for the page."""
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    modes = {}
    for name, cfg in report.get("configurations", {}).items():
        modes[name] = {
            "label": LABELS.get(name, name),
            "overall": cfg["overall"],
            "by_style": {
                style: {"n": s["n"], "hit@10": s["hit@10"]} for style, s in cfg.get("by_style", {}).items()
            },
            "p50_ms": cfg.get("p50_ms"),
            "p95_ms": cfg.get("p95_ms"),
        }
    return {"n_queries": report.get("n_queries"), "n_movies": report.get("n_movies"), "modes": modes}


def _snippet(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 1].rstrip() + "…"


def _item(rank: int, r: dict) -> dict:
    score = r.get("crossencoder_score", r.get("score"))
    return {
        "rank": rank,
        "id": r.get("id"),
        "title": r.get("title", ""),
        "snippet": _snippet(r.get("document") or r.get("description") or ""),
        "score": None if score is None else round(float(score), 4),
    }


def _json(start_response, status: str, body: dict) -> list[bytes]:
    data = json.dumps(body).encode()
    start_response(
        status,
        [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(data))),
            ("Cache-Control", "no-store"),
            ("X-Content-Type-Options", "nosniff"),
        ],
    )
    return [data]


def make_app(searchers: dict[str, Searcher], eval_summary: dict | None = None):
    available = [m for m in MODES if m in searchers]
    summary = eval_summary if eval_summary is not None else load_eval_summary()

    def app(environ, start_response) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        if method != "GET":
            return _json(start_response, "405 Method Not Allowed", {"error": "GET only"})

        if path == "/":
            html = INDEX_HTML.read_bytes()
            start_response(
                "200 OK",
                [
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(html))),
                    (
                        "Content-Security-Policy",
                        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
                    ),
                    ("X-Content-Type-Options", "nosniff"),
                ],
            )
            return [html]

        if path == "/api/config":
            return _json(
                start_response,
                "200 OK",
                {"modes": [{"id": m, "label": LABELS[m]} for m in available], "evals": summary},
            )

        if path == "/api/search":
            params = parse_qs(environ.get("QUERY_STRING", ""))
            query = (params.get("q", [""])[0]).strip()
            if not query:
                return _json(start_response, "400 Bad Request", {"error": "q is required"})
            if len(query) > MAX_QUERY_CHARS:
                return _json(
                    start_response,
                    "400 Bad Request",
                    {"error": f"q is longer than {MAX_QUERY_CHARS} characters"},
                )
            try:
                k = int(params.get("k", ["10"])[0])
            except ValueError:
                return _json(start_response, "400 Bad Request", {"error": "k must be an integer"})
            if not 1 <= k <= MAX_K:
                return _json(start_response, "400 Bad Request", {"error": f"k must be between 1 and {MAX_K}"})
            requested = [m for m in params.get("modes", [",".join(available)])[0].split(",") if m]
            unknown = [m for m in requested if m not in available]
            if unknown:
                return _json(
                    start_response, "400 Bad Request", {"error": f"unavailable modes: {', '.join(unknown)}"}
                )

            results = {}
            for mode in requested:
                started = time.perf_counter()
                try:
                    hits = searchers[mode](query, k)
                except Exception as exc:  # noqa: BLE001 - one broken mode should not blank the page
                    print(f"{mode} failed: {exc!r}", file=sys.stderr)
                    results[mode] = {"error": "search failed", "ms": None, "items": []}
                    continue
                ms = (time.perf_counter() - started) * 1000
                results[mode] = {
                    "ms": round(ms, 1),
                    "items": [_item(i + 1, h) for i, h in enumerate(hits[:k])],
                }
            return _json(start_response, "200 OK", {"query": query, "k": k, "results": results})

        return _json(start_response, "404 Not Found", {"error": "not found"})

    return app


def build_searchers(modes: list[str]) -> dict[str, Searcher]:
    sys.path.insert(0, str(ROOT / "cli"))
    from lib.keyword_search import InvertedIndex

    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        idx.build()
        idx.save()
    searchers: dict[str, Searcher] = {"bm25": lambda q, k: idx.bm25_search(q, k)}
    if set(modes) - {"bm25"}:
        from lib.hybrid_search import HybridSearch
        from lib.reranking import rerank
        from lib.search_utils import RRF_K, SEARCH_MULTIPLIER, load_movies

        hs = HybridSearch(load_movies())
        searchers.update(
            {
                "semantic": lambda q, k: hs.semantic_search.search_chunks(q, k),
                "rrf": lambda q, k: hs.rrf_search(q, k=RRF_K, limit=k),
                "rrf_cross_encoder": lambda q, k: rerank(
                    q, hs.rrf_search(q, k=RRF_K, limit=k * SEARCH_MULTIPLIER), "cross_encoder", k
                ),
            }
        )
    return {m: s for m, s in searchers.items() if m in modes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--modes", default=",".join(MODES), help=f"comma-separated subset of {', '.join(MODES)}"
    )
    args = parser.parse_args(argv)
    modes = [m for m in args.modes.split(",") if m]
    bad = [m for m in modes if m not in MODES]
    if bad:
        parser.error(f"unknown modes: {', '.join(bad)}")

    from wsgiref.simple_server import make_server

    started = time.perf_counter()
    app = make_app(build_searchers(modes))
    print(f"loaded {', '.join(modes)} in {time.perf_counter() - started:.1f}s")
    with make_server(args.host, args.port, app) as httpd:
        print(f"Bingo demo on http://{args.host}:{args.port}")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
