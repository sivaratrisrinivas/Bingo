# Eval audit (eval-audit skill, 2026-10-03)

Routed by evals-start: an eval pipeline exists (known-item queries, four search modes, CI gate), so `eval-audit`. Findings by impact.

## 1. Error analysis

### Metrics existed, but nobody had read the misses
**Status:** Fixed in this branch.
I read 30 randomly sampled BM25 misses on dev (seed 0, out of 77 dev misses) next to the target movie's plot and the top 3 results:

| Failure | Count of 30 |
|---|---:|
| Misspelling hits the one distinctive word ("rmoanian ciatdel") | 14 |
| Vague or paraphrased memory, no shared words with the plot | 10 |
| Hyphenated words in query or plot ("car-accident-decapitation-committee", "post-war", "11-year-old") | 3 |
| Query too generic for one movie ("mythic warrior city adventure") | 2 |
| Query not supported by the plot text (Camille's plot is a junk note) | 1 |

The hyphen failures were a tokenizer bug: `preprocess_text` deleted hyphens, so "post-war" became the token "postwar", which no query matches. Fix: split on hyphens, slashes and dashes, and keep dropping other punctuation. A unit test was added. BM25 after the fix: dev hit@10 0.841 -> 0.863, test hit@10 0.858 -> 0.863, test MRR@10 0.701 -> 0.698, test hit@1 0.626 -> 0.618. hit@10 improves; MRR on test is flat.

Misspellings (fuzzy matching) and vague queries (semantic or rerank) are the next real product work. The rerank mode already helps most there (see README).

## 5. Labeled data

### Queries are LLM-generated, and a few are unanswerable or ambiguous
**Status:** Problem exists (partly).
3 of the 30 misses read were label problems rather than search problems (2 too generic, 1 not supported by the plot text). That is about 10% of misses, so true hit@10 is a little higher than reported. Descriptive queries were written from plot text and favor BM25.
**Fix:** replace generated queries with logged real searches as they arrive (README "Monitoring path"). Drop or fix queries a reviewer marks unanswerable.

## 2. Evaluator design

**Status:** OK. Each query has one known target and is scored by exact rank (hit@k, MRR), computed by code. No LLM judge and no similarity metric used as correctness.

## 3. Judge validation

**Status:** Not applicable (no LLM judge in this eval). The RAG answer path (Gemini) has no eval at all. It needs `GEMINI_API_KEY` and traces first, and then `error-discovery`.

## 4. Human review

**Status:** Problem exists. Query labels come from the generator and were reviewed only through the 30 misses above. A person should skim a random 50 queries for answerability.

## 6. Pipeline hygiene

**Status:** OK. CI rebuilds the BM25 index and gates on dev/test floors. The full four-mode run is a manual job and must be re-run after tokenizer or model changes (done for this change, see README).
