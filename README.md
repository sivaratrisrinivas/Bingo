# Bingo

A movie search engine that combines BM25 ranking and semantic search to find relevant movies from large datasets.

---

## What is Bingo?

Bingo is a command-line tool that searches through thousands of movie descriptions using two complementary approaches:

- **BM25 Search**: Smart keyword matching that considers word frequency, rarity, and document length
- **Semantic Search**: AI-powered understanding that finds conceptually similar movies even without exact keyword matches

Together, these methods can be combined for even better results.

---

## Why Two Approaches?

**Traditional keyword search** only finds exact word matches. **BM25** is smarter—it considers how often words appear, how rare they are, and adjusts for document length. **Semantic search** goes further by understanding meaning and context, finding movies that are similar in concept even if they don't share keywords.

---

## Results

Retrieval quality on `data/golden_dataset.json` (10 queries, 5000 movies, k=5). Re-measured 2026-10-02 on an Intel Xeon (8 CPUs, 15.64 GB RAM, Python 3.13.5) after the BM25 rewrite. Embedding model: `all-MiniLM-L6-v2`. Reranker: `cross-encoder/ms-marco-TinyBERT-L2-v2`.

| Configuration | Precision@5 | Recall@5 | F1 | p95 query latency (ms) |
|---|---|---|---|---|
| BM25 | 0.4200 | 0.5407 | 0.3915 | 4.9 |
| Semantic | 0.3200 | 0.4606 | 0.3088 | 1818.2 |
| RRF | 0.4200 | 0.5299 | 0.3906 | 1216.7 |
| RRF + cross-encoder | 0.3600 | 0.4824 | 0.3357 | 1775.2 |

Changes from the 2026-08-24 run (BM25 0.3800 / 0.5157 / 0.3607, p95 3419.2 ms): BM25 now scores from precomputed term statistics instead of re-reading every document per query, so p95 fell from about 3.4 s to 5 ms. The quality gain comes from a bug fix: the old code stemmed query terms twice inside `bm25()`, which changed the "zombie apocalypse" results. In an old-versus-new check on 15 queries, the other 14 returned identical rankings, and `tests/` checks the fast scorer against a brute-force BM25. Semantic and reranker latency is higher than in August because this run shared the machine with other jobs (load average about 19), so treat those latency numbers as noisy. Index build this run: 2022.3 s (BM25 5.0 s, chunk embeddings 2017.3 s on the loaded machine).

The 2026-10-03 hyphen tokenizer fix was not re-measured on this table. None of the 10 golden queries contains a hyphen, and a rerun after the fix gave identical per-query precision and recall for the configuration `eval/run_retrieval_metrics.py` stores. With 10 queries, one query moves Precision@5 by 0.02 or more, so the golden table is a sanity check. The larger known-item eval below is the main quality measure.

The golden set lists `Død snø` as relevant for the zombie query. That title is not in `movies.json`, so no method can retrieve it.

Reproduce from a clean checkout (downloads `data/movies.json` if missing):

```bash
uv run python eval/run_retrieval_metrics.py
```

This writes `eval/results.json` and prints the table.

## Evals

### Known-item search (main eval)

A known-item query is what someone types when they remember a movie but not its title. `eval/known_item/` holds 948 generated queries for 237 movies sampled across description lengths (the target was 300 movies; generation stopped at 237 when the Groq free-tier limit ran out):

- An LLM (gpt-oss-safeguard-20b on Groq) wrote three queries per movie: keyword, descriptive, and vague. Code added a misspelled copy of each keyword query.
- Code dropped 114 queries that contain a word from the title, since those are too easy. 834 queries over 233 movies remain.
- Each query has exactly one correct movie. Split by movie id: dev 496 queries, test 452 (before title-leak removal).
- These queries are generated, not real user searches. Bingo has no query logs yet. Descriptive queries were written from the plot text, so they favor word matching, which helps BM25.

Results re-measured on 2026-10-03 after the hyphen tokenizer fix (`uv run python eval/known_item/run_known_item.py`). Before the fix, BM25 was dev 0.841 / test 0.858 hit@10, test MRR@10 0.701.

| Configuration | dev hit@10 | test hit@10 | test hit@1 | test MRR@10 | p95 latency |
|---|---|---|---|---|---|
| BM25 | 0.863 | 0.863 | 0.618 | 0.698 | 33 ms |
| Semantic | 0.573 | 0.626 | 0.371 | 0.448 | 1136 ms |
| RRF | 0.811 | 0.837 | 0.571 | 0.656 | 681 ms |
| RRF + cross-encoder | 0.894 | 0.916 | 0.697 | 0.771 | 856 ms |

hit@10 by query style:

| Configuration | keyword | misspelled keyword | descriptive | vague |
|---|---|---|---|---|
| BM25 | 0.948 | 0.761 | 0.955 | 0.798 |
| Semantic | 0.667 | 0.464 | 0.714 | 0.557 |
| RRF | 0.857 | 0.689 | 0.965 | 0.793 |
| RRF + cross-encoder | 0.962 | 0.820 | 0.975 | 0.867 |

What this says in plain words:

- The cross-encoder rerank is the best mode: about 9 in 10 queries find the movie in the top 10.
- Plain semantic search is the weakest. Fusing it with BM25 (RRF) is slightly worse than BM25 alone on this set.
- Misspellings and vague memories are the hard cases. Reading 30 BM25 misses also showed hyphenated words failing: the tokenizer deleted hyphens, so "post-war" became "postwar". That is fixed (see `eval/known_item/AUDIT.md`).
- Latency was measured on a shared, busy machine, so treat it as rough.

CI rebuilds the BM25 index and fails if BM25 hit@10 or MRR@10 drops below `eval/known_item/baseline.json`. The full four-mode run is a manual workflow job because it needs the embedding model and takes a long time on CPU.

### Monitoring path

Log each search query with the mode, the top 10 ids, and whether the user clicked a result. Each week, sample about 50 queries with no click in the top 10, label the intended movie by hand when it can be found, and add them to the query set as real cases.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sivaratrisrinivas/Bingo
cd Bingo

# Install dependencies (uv recommended)
uv sync --frozen

# Or using pip
pip install -e .

# Optional: For RAG and query enhancement, add to .env:
# GEMINI_API_KEY=your_api_key_here
```

---

## Data Setup

Place your movie dataset at `data/movies.json`:

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Movie Title",
      "description": "Movie description..."
    }
  ]
}
```

Also ensure `data/stopwords.txt` exists (one stopword per line).

---

## How It Works

### Keyword Search (BM25)
1. **Index Building**: Creates an inverted index (like a book's index, but for every word)
2. **Text Processing**: Cleans and stems words (e.g., "running" → "run")
3. **Smart Scoring**: Uses BM25 algorithm to rank results by relevance

### Semantic Search
1. **Model Loading**: Loads pre-trained sentence transformer model (all-MiniLM-L6-v2)
2. **Text Embedding**: Converts queries and descriptions into high-dimensional vectors
3. **Similarity Matching**: Finds movies with similar meaning using cosine similarity
4. **Chunking**: Splits long text into manageable chunks for better processing

### Hybrid Search
1. **Dual Query**: Runs both BM25 and semantic search simultaneously
2. **Result Merging**: Combines results from both approaches
3. **Ranking**: Sorts merged results by relevance score

---

## Quick Start

### 1. Setup

```bash
# Build search index (required first step)
python cli/keyword_search_cli.py build

# Precompute semantic embeddings (recommended)
python cli/semantic_search_cli.py embed_chunks
```

### 2. Basic Searches

```bash
# Keyword search
python cli/keyword_search_cli.py bm25search "romantic comedy" --limit 10

# Semantic search
python cli/semantic_search_cli.py search "space adventure" --limit 5

# Hybrid search (best results)
python cli/hybrid_search_cli.py rrf-search "superhero action movie" --limit 10
```

### 3. Advanced Features

```bash
# Query enhancement (spell correction)
python cli/hybrid_search_cli.py rrf-search "briish bear" --enhance spell

# Query rewriting (makes vague queries searchable)
python cli/hybrid_search_cli.py rrf-search "bear movie that gives me the lulz" --enhance rewrite

# Reranking (improves result quality)
python cli/hybrid_search_cli.py rrf-search "action movie" --rerank-method cross_encoder --limit 25

# Multimodal query rewriting (image + text)
python cli/describe_image_cli.py --image data/paddington.jpeg --query "find movies like this"

# Verify image embedding generation
python cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg

# Search movies using image
python cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

### 4. RAG (AI-Powered Answers)

```bash
# Generate comprehensive answer
python cli/augmented_generation_cli.py rag "movies about time travel"

# Summarize results
python cli/augmented_generation_cli.py summarize "action movies" --limit 10

# Answer with citations
python cli/augmented_generation_cli.py citations "time travel movies" --limit 10

# Casual Q&A
python cli/augmented_generation_cli.py question "What are some good horror movies?" --limit 10
```

---

## Commands Reference

### Keyword Search (`keyword_search_cli.py`)

| Command | Description |
|---------|-------------|
| `build` | Create search index from movie data |
| `search <query>` | Basic keyword search (exact matching) |
| `bm25search <query>` | BM25-ranked search |
| `tf <doc_id> <term>` | Get term frequency |
| `idf <term>` | Get inverse document frequency |
| `tfidf <doc_id> <term>` | Get TF-IDF score |
| `bm25idf <term>` | Get BM25-IDF score |
| `bm25tf <doc_id> <term>` | Get BM25 term frequency |

### Semantic Search (`semantic_search_cli.py`)

| Command | Description |
|---------|-------------|
| `verify` | Verify embedding model is loaded |
| `verify_embeddings` | Verify movie embeddings and show stats |
| `search <query>` | Semantic similarity search |
| `embed_chunks` | Precompute sentence-chunk embeddings |
| `search_chunked <query> [--limit <int>]` | Search using chunked embeddings |
| `chunk <text> [--chunk-size <int>] [--overlap <int>]` | Split into word chunks |
| `semantic_chunk <text> [--max-chunk-size <int>] [--overlap <int>]` | Split into sentence chunks |

### Hybrid Search (`hybrid_search_cli.py`)

| Command | Description |
|---------|-------------|
| `hybrid_search <query> [--limit <int>]` | Combine BM25 + semantic results |
| `weighted-search <query> [--alpha <float>] [--limit <int>]` | Weighted combination (alpha: 0.0=semantic, 1.0=BM25, default 0.5) |
| `rrf-search <query> [--k <int>] [--limit <int>] [--enhance <method>] [--rerank-method <method>] [--evaluate]` | Reciprocal Rank Fusion |
| `normalize <scores...>` | Min-max normalization to [0, 1] |

**RRF Search Options:**
- `--enhance`: `spell` (correction), `rewrite` (query rewriting), `expand` (synonyms)
- `--rerank-method`: `individual` (LLM per result), `batch` (single LLM call), `cross_encoder` (local model)
- `--evaluate`: Rate results 0-3 for relevance

### RAG (`augmented_generation_cli.py`)

| Command | Description |
|---------|-------------|
| `rag <query>` | Search + generate comprehensive answer |
| `summarize <query> [--limit <int>]` | Generate concise 3-4 sentence summary |
| `citations <query> [--limit <int>]` | Generate answer with [1], [2] citations |
| `question <question> [--limit <int>]` | Casual, conversational Q&A |

**Note**: RAG commands require `GEMINI_API_KEY` in `.env` or environment variables.

### Evaluation (`evaluation_cli.py`)

| Command | Description |
|---------|-------------|
| `evaluation_cli.py [--limit <int>]` | Evaluate search performance using golden dataset (precision@k, recall@k, F1) |

### Multimodal Query Rewriting (`describe_image_cli.py`)

| Command | Description |
|---------|-------------|
| `--image <path>` | Path to image file (required) |
| `--query <text>` | Text query to rewrite based on image (required) |

**Note**: Requires `GEMINI_API_KEY` in `.env` or environment variables.

### Multimodal Search (`multimodal_search_cli.py`)

| Command | Description |
|---------|-------------|
| `verify_image_embedding <image>` | Verify image embedding generation (512-dimensional CLIP embeddings) |
| `image_search <image>` | Search for movies using image similarity (returns top 5 matches) |

---

## Technical Details

### Keyword Search
- **Library**: NLTK for text processing and stemming
- **Algorithm**: BM25 with parameters k1=1.5, b=0.75
- **Caching**: Processed data cached for performance
- **Features**: Custom search limits, detailed TF-IDF/BM25 scoring

### Semantic Search
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Embeddings**: 384-dimensional vectors
- **Cache**: `cache/movie_embeddings.npy` (auto-rebuilt if dataset changes)
- **Context**: Combines titles + descriptions for richer understanding
- **Defaults**: CPU device, search limit 5
- **Chunking**: Default size 200 words, overlap 1; sentence chunks default to 4 sentences

**Precompute chunked embeddings:**
```bash
python cli/semantic_search_cli.py embed_chunks
# Expected: "Generated 72909 chunked embeddings"
```

**Rebuild cache:**
```bash
rm cache/chunk_embeddings.npy cache/chunk_metadata.json
python cli/semantic_search_cli.py embed_chunks
```

### Hybrid Search

**Weighted Hybrid Search**
- Formula: `hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic`
- Searches top 2500 from each method, normalizes scores, combines, returns top `--limit`

**Reciprocal Rank Fusion (RRF)**
- Formula: `rrf_score = 1 / (k + rank)` (default k=60)
- Uses ranks (position) rather than scores, robust to different score distributions
- Documents in both result sets have RRF scores summed

**Reranking Methods:**
- `individual`: LLM scores each result (0-10 scale), slower, more API calls
- `batch`: LLM reranks all results in single call, faster, fewer API calls
- `cross_encoder`: Local neural network model, fastest, no API key required

**Query Enhancement:**
- `spell`: Corrects spelling errors
- `rewrite`: Rewrites vague queries to be more searchable
- `expand`: Expands queries with synonyms and related terms

**Result Evaluation:**
- Rates results 0-3 (3=Highly relevant, 2=Relevant, 1=Marginally relevant, 0=Not relevant)

### RAG (Retrieval Augmented Generation)
- **Search**: RRF (k=60), retrieves `limit * 5` results (default: 25)
- **Model**: Gemini 2.0 Flash
- **Context**: Movie titles + descriptions
- **Commands**:
  - `rag`: Comprehensive answers
  - `summarize`: 3-4 sentence summaries
  - `citations`: Answers with [1], [2] source citations
  - `question`: Casual, conversational Q&A
- **Output**: Search result titles → AI-generated answer/summary

### Evaluation
- **Dataset**: `data/golden_dataset.json` (queries + expected relevant titles)
- **Metrics**: Precision@k, Recall@k, F1 score
- **Method**: RRF search (k=60) for retrieval

### Multimodal Query Rewriting
- **Model**: Gemini 2.0 Flash
- **Input**: Image file + text query
- **Process**: Analyzes image content, synthesizes with text query, rewrites for movie search
- **Output**: Rewritten query optimized for movie database search + token usage
- **MIME Types**: Auto-detects image format (JPEG, PNG, etc.), defaults to JPEG

### Multimodal Search
- **Model**: CLIP ViT-B-32 (sentence-transformers)
- **Embeddings**: 512-dimensional vectors
- **Device**: CPU by default (CUDA disabled to prevent GPU mismatches)
- **Purpose**: Generate image embeddings for visual similarity search
- **Search**: Compares image embeddings with movie title+description text embeddings using cosine similarity
- **Format**: Movies formatted as "title: description" for CLIP text encoding

### Performance
- **Device**: CPU by default (CUDA disabled to prevent GPU mismatches)
- **Caching**: Indexes and embeddings cached for fast repeated searches
- **Defaults**: Search limit 5, RRF k=60, search multiplier 5

---

## Output Format

**Chunked semantic search:**
```
1. The Incredibles (score: 0.8123)
   A family of undercover superheroes, while trying to live the quiet suburban life...
```

**RRF search:**
```
Movie Title (RRF score: 0.0234, BM25 rank: 3, Semantic rank: 5)
Description preview...
```

**RAG:**
```
Search Results:
  - Movie Title 1
  - Movie Title 2

RAG Response:
[AI-generated answer]
```

**Multimodal query rewriting:**
```
Rewritten query: Paddington Bear family-friendly animated adventure movies
Total tokens:    1234
```

**Multimodal search:**
```
Image search results for: data/paddington.jpeg
============================================================
1. Paddington (similarity: 0.309)
   A young Peruvian bear travels to London in search of a home...
2. Paddington 2 (similarity: 0.285)
   Paddington, now happily settled with the Brown family...
```

---

## Notes

- All search methods require the keyword index (`build` command)
- Hybrid search and RAG require chunked embeddings (`embed_chunks` command)
- RAG and multimodal query rewriting require `GEMINI_API_KEY` for AI features
- Cache files are automatically rebuilt if dataset size changes
- The system runs on CPU by default to avoid GPU compatibility issues
