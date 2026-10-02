import os
import re
import time
from typing import Optional

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .reranking import rerank
from .query_enhancement import enhance_query
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    load_movies,
    format_search_result,
)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        if not self.idx.docmap:
            self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        fused = rrf_combine_search_results(bm25_results, semantic_results, k) 
        return fused[:limit]

def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores using min-max normalization to range [0, 1]."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    # If all scores are the same, return all 1.0
    if min_score == max_score:
        return [1.0] * len(scores)
    
    # Min-max normalization: (score - min) / (max - min)
    normalized_scores = []
    for score in scores:
        normalized_scores.append((score - min_score) / (max_score - min_score))

    return normalized_scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }

def rrf_score(rank: int, k: int = 60) -> float:
    """Calculate RRF score for a given rank."""
    return 1 / (k + rank)


def rrf_combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], k: int = 60
) -> list[dict]:
    """
    Combine BM25 and semantic search results using Reciprocal Rank Fusion (RRF).
    
    Args:
        bm25_results: List of BM25 search results
        semantic_results: List of semantic search results
        k: RRF constant (default 60)
    
    Returns:
        List of results sorted by RRF score in descending order
    """
    # Map document IDs to documents and their ranks/scores
    doc_data = {}
    
    # Process BM25 results with ranks (1-indexed)
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in doc_data:
            doc_data[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "document": result.get("document", ""),
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": 0.0,
            }
        # Calculate and add BM25 RRF score
        doc_data[doc_id]["rrf_score"] += rrf_score(rank, k)
    
    # Process semantic results with ranks (1-indexed)
    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in doc_data:
            doc_data[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "document": result.get("document", ""),
                "bm25_rank": None,
                "semantic_rank": rank,
                "rrf_score": 0.0,
            }
        else:
            doc_data[doc_id]["semantic_rank"] = rank
        # Calculate and add semantic RRF score (sum if doc appears in both)
        doc_data[doc_id]["rrf_score"] += rrf_score(rank, k)
    
    # Convert to list and sort by RRF score descending
    results = []
    for doc_id, data in doc_data.items():
        result = format_search_result(
            doc_id=data["id"],
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        results.append(result)
    
    return sorted(results, key=lambda x: x["score"], reverse=True) 

def rrf_search_command(
    query: str, k: int = RRF_K, enhance: Optional[str] = None, rerank_method: Optional[str] = None, limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query
    
    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = searcher.rrf_search(query, k, search_limit)

    reranked = False
    if rerank_method:
        results = rerank(query, results, method=rerank_method, limit=limit)
        reranked = True

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "results": results,
    }