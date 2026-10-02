import os
import math
import string
import pickle
from collections import defaultdict, Counter
from functools import lru_cache

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_K1,
    DEFAULT_SEARCH_LIMIT,
    SCORE_PRECISION,
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    load_movies,
    load_stopwords,
    load_stopword_set,
    format_search_result,
)

class InvertedIndex: 
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.term_frequencies = defaultdict(Counter)
        self.docmap: dict[int, dict] = {}
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self._avg_doc_length: float | None = None

    def build(self) -> None: 
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie["title"]} {movie["description"]}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def save(self) -> None: 
        os.makedirs(CACHE_DIR, exist_ok= True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f: 
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
        self._avg_doc_length = None

    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        if self._avg_doc_length is not None:
            return self._avg_doc_length
        total_length = 0.0
        for length in self.doc_lengths.values():
            total_length += length
        self._avg_doc_length = total_length / len(self.doc_lengths)
        return self._avg_doc_length
    
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: str, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Expected a single-token term")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1.0)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        saturated_tf_score = (tf * (k1 + 1)) / (tf + k1)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        saturated_tf_score = tf_component
        return saturated_tf_score

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int) -> list[dict]:
        """Score only documents that contain a query term.

        Same ranking as scoring every document: documents with no query term
        score 0 and keep their corpus order after the scored ones.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if limit < 0:
            raise ValueError("limit must be >= 0")
        query_tokens = tokenize_text(query)
        doc_count = len(self.docmap)
        avg_len = self.__get_avg_doc_length()
        k1, b = BM25_K1, BM25_B

        idf = {}
        for token in set(query_tokens):
            n = len(self.index.get(token, ()))
            idf[token] = math.log((doc_count - n + 0.5) / (n + 0.5) + 1.0)

        candidates = set()
        for token in query_tokens:
            candidates.update(self.index.get(token, ()))

        scores = {}
        for doc_id in self.docmap:
            if doc_id not in candidates:
                continue
            tfs = self.term_frequencies[doc_id]
            norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_len) if avg_len > 0 else 1.0
            score = 0.0
            for token in query_tokens:
                tf = tfs[token]
                score += ((tf * (k1 + 1)) / (tf + k1 * norm)) * idf[token]
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_docs) < limit:
            for doc_id in self.docmap:
                if doc_id not in scores:
                    sorted_docs.append((doc_id, 0.0))
                    if len(sorted_docs) >= limit:
                        break

        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)
        return results
        





        

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            if not doc:
                continue
            results.append(doc)
            if len(results) >= limit:
                return results
    return results


def preprocess_text(text: str) -> str: 
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens: 
        for title_token in title_tokens: 
            if query_token in title_token: 
                return True
    return False


def tokenize_text(text: str) -> list[str]: 
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens: 
        if token: 
            valid_tokens.append(token)
    stopwords = load_stopword_set()
    return [_stem(word) for word in valid_tokens if word not in stopwords]


_STEMMER = PorterStemmer()


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _STEMMER.stem(word)

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    bm25_tf_score = idx.get_bm25_tf(doc_id, term, k1, b)
    return bm25_tf_score

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)

