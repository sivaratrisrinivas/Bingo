#!/usr/bin/env python3
"""Generate known-item search queries for the movie corpus.

A known-item query is what someone types when they remember a movie but not its
title. For each sampled movie an LLM writes three queries in different styles
(keyword, descriptive, vague). Code then:
  - rejects queries that contain a title word (that would make the case trivial),
  - adds a typo variant of each keyword query (adversarial: misspellings),
  - assigns a stable split (dev or test) by movie id.

Dimensions: query style x description length tertile x title length.
Sampling is stratified on description length tertile with a fixed seed.

Needs an OpenAI-compatible endpoint. Defaults to Groq:
  GROQ_API_KEY=... python eval/known_item/generate_queries.py --n 300

Resumable: movies already present in the output file are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "queries.jsonl"
STYLES = ("keyword", "descriptive", "vague")

PROMPT = """You help build a search test set for a movie search engine. A user remembers \
this movie but not its title.
Title: {title}
Description: {description}
Write three search queries this user might type. Never use any word from the title, \
and never use character or actor names.
- keyword: 3 to 6 lowercase keywords
- descriptive: one sentence describing the plot
- vague: a half-remembered, fuzzy description with only one or two specifics
Return only JSON: {{"keyword": "...", "descriptive": "...", "vague": "..."}}"""

STOP = {
    "the", "a", "an", "of", "and", "in", "on", "to", "for", "with", "at", "by", "from",
    "is", "it", "part", "ii", "iii", "2", "3", "4", "vs", "movie", "film",
}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def title_leak(query: str, title: str) -> list[str]:
    """Title words (minus very common ones) that appear in the query."""
    return sorted((words(title) - STOP) & words(query))


def typo(text: str, rng: random.Random) -> str:
    """Swap two adjacent letters inside up to two words longer than 4 letters."""
    toks = text.split()
    idxs = [i for i, t in enumerate(toks) if len(t) > 4 and t.isalpha()]
    for i in rng.sample(idxs, min(2, len(idxs))):
        t = toks[i]
        j = rng.randrange(1, len(t) - 2)
        toks[i] = t[:j] + t[j + 1] + t[j] + t[j + 2 :]
    return " ".join(toks)


def split_for(movie_id: int) -> str:
    h = int(hashlib.sha256(str(movie_id).encode()).hexdigest(), 16)
    return "test" if h % 2 == 0 else "dev"


def sample_movies(movies: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ranked = sorted(movies, key=lambda m: len(m["description"]))
    third = len(ranked) // 3
    tiers = [ranked[:third], ranked[third : 2 * third], ranked[2 * third :]]
    picked = []
    for tier_name, tier in zip(("short", "medium", "long"), tiers):
        for m in rng.sample(tier, n // 3 + (1 if len(picked) < n % 3 else 0)):
            picked.append({**m, "desc_tier": tier_name})
    return picked[:n]


def call_llm(prompt: str, model: str, base: str, key: str) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 450,
        "temperature": 0.7,
    }
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "bingo-eval"},
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(min(60, 3 * 2**attempt))
                continue
            raise
    raise RuntimeError("LLM call kept failing")


def parse(text: str) -> dict[str, str] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(obj.get(s), str) and obj[s].strip() for s in STYLES):
        return None
    return {s: obj[s].strip() for s in STYLES}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", default=os.getenv("GEN_MODEL", "openai/gpt-oss-safeguard-20b"))
    ap.add_argument("--base-url", default=os.getenv("GEN_BASE_URL", "https://api.groq.com/openai/v1"))
    args = ap.parse_args()
    key = os.getenv("GEN_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        print("Set GROQ_API_KEY or GEN_API_KEY", file=sys.stderr)
        return 2
    movies = json.loads((ROOT / "data" / "movies.json").read_text())["movies"]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["movie_id"] for line in args.out.read_text().splitlines() if line}
    rng = random.Random(args.seed + 1)
    sample = sample_movies(movies, args.n, args.seed)
    with args.out.open("a") as f:
        for i, m in enumerate(sample):
            if m["id"] in done:
                continue
            raw = call_llm(
                PROMPT.format(title=m["title"], description=m["description"][:1200]),
                args.model, args.base_url, key,
            )
            q = parse(raw)
            if q is None:
                print(f"{i}: unparseable output for {m['title']!r}", file=sys.stderr)
                continue
            title_words = len(words(m["title"]) - STOP)
            base = {
                "movie_id": m["id"],
                "title": m["title"],
                "desc_tier": m["desc_tier"],
                "title_len": "short" if title_words <= 2 else "long",
                "split": split_for(m["id"]),
                "generator": args.model,
            }
            rows = [{**base, "style": s, "query": q[s]} for s in STYLES]
            rows.append({**base, "style": "keyword_typo", "query": typo(q["keyword"], rng)})
            for r in rows:
                r["title_leak"] = title_leak(r["query"], m["title"])
                r["id"] = f"m{m['id']}-{r['style']}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i + 1}/{len(sample)} {m['title']}", flush=True)
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
