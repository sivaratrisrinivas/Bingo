#!/usr/bin/env python3
"""Known-item retrieval eval: can each search mode find the one movie a user means?

Reads eval/known_item/queries.jsonl (see generate_queries.py). Each query has exactly
one target movie id. Metrics per configuration, split, and query style:
  hit@1, hit@5, hit@10  (share of queries whose target is in the top k)
  MRR@10                (mean of 1/rank of the target, 0 if not in top 10)

Queries that leak a title word are excluded and counted.

  python eval/known_item/run_known_item.py --configs bm25          # fast, used in CI
  python eval/known_item/run_known_item.py                         # all configs (builds
                                                                   # embeddings, ~6 min CPU)
  python eval/known_item/run_known_item.py --check                 # exit 1 on regression
                                                                   # vs baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "cli"))

ALL_CONFIGS = ("bm25", "semantic", "rrf", "rrf_cross_encoder")
KS = (1, 5, 10)


def load_queries(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows


def rank_of(target: int, results: list[dict]) -> int | None:
    for i, r in enumerate(results, start=1):
        if r["id"] == target:
            return i
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out: dict = {"n": n}
    if n == 0:
        return out
    for k in KS:
        out[f"hit@{k}"] = round(sum(r is not None and r <= k for r in ranks) / n, 4)
    out["mrr@10"] = round(sum(1 / r for r in ranks if r is not None and r <= 10) / n, 4)
    return out


def make_retrievers(configs: list[str]) -> dict:
    retrievers = {}
    if configs == ["bm25"]:
        from lib.keyword_search import InvertedIndex

        idx = InvertedIndex()
        try:
            idx.load()
        except FileNotFoundError:
            idx.build()
        retrievers["bm25"] = lambda q: idx.bm25_search(q, 10)
        return retrievers

    from lib.hybrid_search import HybridSearch
    from lib.reranking import rerank
    from lib.search_utils import RRF_K, SEARCH_MULTIPLIER, load_movies

    hs = HybridSearch(load_movies())
    table = {
        "bm25": lambda q: hs._bm25_search(q, 10),
        "semantic": lambda q: hs.semantic_search.search_chunks(q, 10),
        "rrf": lambda q: hs.rrf_search(q, k=RRF_K, limit=10),
        "rrf_cross_encoder": lambda q: rerank(
            q, hs.rrf_search(q, k=RRF_K, limit=10 * SEARCH_MULTIPLIER), "cross_encoder", 10
        ),
    }
    return {c: table[c] for c in configs}


def evaluate(queries: list[dict], configs: list[str]) -> dict:
    usable = [q for q in queries if not q["title_leak"]]
    report: dict = {
        "n_queries_total": len(queries),
        "n_excluded_title_leak": len(queries) - len(usable),
        "n_queries": len(usable),
        "n_movies": len({q["movie_id"] for q in usable}),
        "configurations": {},
    }
    retrievers = make_retrievers(configs)
    misses: dict = {}
    for name, fn in retrievers.items():
        by: dict[tuple[str, str], list] = defaultdict(list)
        lat = []
        conf_misses = []
        for q in usable:
            t = time.perf_counter()
            results = fn(q["query"])
            lat.append((time.perf_counter() - t) * 1000)
            r = rank_of(q["movie_id"], results)
            for key in [("split", q["split"]), ("style", q["style"]), ("desc_tier", q["desc_tier"]),
                        ("all", "all")]:
                by[key].append(r)
            by[("split_style", f"{q['split']}/{q['style']}")].append(r)
            if r is None:
                conf_misses.append(q["id"])
        lat.sort()
        report["configurations"][name] = {
            "overall": summarize(by[("all", "all")]),
            "by_split": {k[1]: summarize(v) for k, v in by.items() if k[0] == "split"},
            "by_style": {k[1]: summarize(v) for k, v in by.items() if k[0] == "style"},
            "by_desc_tier": {k[1]: summarize(v) for k, v in by.items() if k[0] == "desc_tier"},
            "by_split_style": {k[1]: summarize(v) for k, v in sorted(by.items())
                               if k[0] == "split_style"},
            "p50_ms": round(lat[len(lat) // 2], 1),
            "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1),
        }
        misses[name] = conf_misses
    report["misses_top10"] = misses
    return report


def check(report: dict, baseline: dict) -> list[str]:
    problems = []
    for cfg, metrics in baseline.items():
        got = report["configurations"].get(cfg)
        if got is None:
            continue
        for split, floor in metrics.items():
            for metric, value in floor.items():
                have = got["by_split"].get(split, {}).get(metric)
                if have is not None and have < value:
                    problems.append(f"{cfg} {split} {metric} {have} < {value}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, default=HERE / "queries.jsonl")
    ap.add_argument("--configs", default=",".join(ALL_CONFIGS))
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(configs) - set(ALL_CONFIGS)
    if unknown:
        ap.error(f"unknown configs: {sorted(unknown)}")
    report = evaluate(load_queries(args.queries), configs)
    print(f"{report['n_queries']} queries over {report['n_movies']} movies "
          f"({report['n_excluded_title_leak']} excluded for title leaks)")
    print(f"{'config':18s} {'split':5s} {'hit@1':>6s} {'hit@5':>6s} {'hit@10':>6s} {'mrr@10':>6s}")
    for cfg, r in report["configurations"].items():
        for split, s in sorted(r["by_split"].items()):
            print(f"{cfg:18s} {split:5s} {s['hit@1']:6.3f} {s['hit@5']:6.3f} {s['hit@10']:6.3f} "
                  f"{s['mrr@10']:6.3f}")
        styles = ", ".join(f"{k} {v['hit@10']:.3f}" for k, v in sorted(r["by_style"].items()))
        print(f"{'':18s} hit@10 by style: {styles}; p95 {r['p95_ms']} ms")
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    if args.check:
        problems = check(report, json.loads((HERE / "baseline.json").read_text()))
        if problems:
            print("REGRESSION: " + "; ".join(problems))
            return 1
        print("known-item gates pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
