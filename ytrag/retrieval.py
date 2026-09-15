"""Hybrid search combining BM25 (lexical) + Vector (semantic) retrieval.

The two retrieval methods fail differently:
- Vector search understands meaning but struggles with exact terminology
- BM25 finds exact keywords but misses semantic similarity

Combining them gives us both signal: "I found the exact words AND understand the meaning."
"""

import math
from collections import Counter
import re

from ytrag.config import TOP_K, BM25_WEIGHT, VECTOR_WEIGHT
from ytrag.index import search as vector_search, get_all_chunks
from ytrag.models import Chunk


def _tokenize(text: str) -> list[str]:
    """Split text into tokens, lowercased and deduplicated."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def _compute_idf_and_avg_len(tokenized_corpus: list[list[str]]) -> tuple[dict[str, float], float]:
    """Compute IDF and exact average document length for a tokenized corpus.

    Args:
        tokenized_corpus: List of token strings for each document

    Returns:
        Tuple of (Dictionary mapping term -> IDF score, Average document length)
    """
    doc_count = len(tokenized_corpus)
    if doc_count == 0:
        return {}, 100.0

    total_tokens = sum(len(doc) for doc in tokenized_corpus)
    avg_length = total_tokens / doc_count

    term_doc_count: dict[str, int] = Counter()
    for doc in tokenized_corpus:
        tokens_set = set(doc)
        for token in tokens_set:
            term_doc_count[token] += 1

    idf = {}
    for term, count in term_doc_count.items():
        # Standard IDF formula: log(N / df)
        idf[term] = math.log(doc_count / (1 + count))

    return idf, avg_length


def bm25_score(
    query_tokens: list[str],
    doc_term_counts: dict[str, int],
    doc_length: int,
    idf: dict[str, float],
    avg_length: float,
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    """Calculate BM25 score for a query against a document.

    BM25 is the standard text retrieval ranking function. It scores based on:
    - Term frequency in the document (saturates to avoid over-weighting common terms)
    - Inverse document frequency (terms that appear in few docs rank higher)
    - Document length normalization (prevents long documents from always scoring high)

    Args:
        query_tokens: Tokenized user question
        doc_term_counts: Term frequencies in the candidate chunk text
        doc_length: Length of the candidate chunk text in tokens
        idf: Pre-computed IDF scores
        avg_length: Average chunk length across the corpus
        k1: Controls term frequency saturation (typically 1.5)
        b: Controls length normalization (0 = no normalization, 1 = full)

    Returns:
        BM25 score (higher is better)
    """
    score = 0.0
    for term in query_tokens:
        term_freq = doc_term_counts.get(term, 0)
        if term_freq == 0:
            continue

        # IDF factor
        term_idf = idf.get(term, 0.0)

        # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
        normalized_length = 1 - b + b * (doc_length / max(avg_length, 1.0))
        score += term_idf * (term_freq * (k1 + 1)) / (term_freq + k1 * normalized_length)

    return score


def bm25_search(
    query: str,
    top_k: int = TOP_K,
    video_id: str | None = None,
) -> list[tuple[Chunk, float]]:
    """BM25 search over the entire corpus (or filtered by video_id).

    Args:
        query: User question
        top_k: Number of results to return
        video_id: Optional filter to specific video

    Returns:
        List of (Chunk, BM25 score) tuples, sorted by score descending
    """
    # Get all chunks (or filtered by video_id)
    chunks = get_all_chunks(video_id)
    if not chunks:
        return []

    # Tokenize once, calculate corpus average length, and get IDF
    tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]
    idf, avg_length = _compute_idf_and_avg_len(tokenized_corpus)

    query_tokens = _tokenize(query)

    # Score each chunk
    scores: list[tuple[Chunk, float]] = []
    for chunk, tokens in zip(chunks, tokenized_corpus):
        doc_term_counts = Counter(tokens)
        score = bm25_score(
            query_tokens=query_tokens,
            doc_term_counts=doc_term_counts,
            doc_length=len(tokens),
            idf=idf,
            avg_length=avg_length,
        )
        scores.append((chunk, score))

    # Sort by score descending and take top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def hybrid_search(
    query: str,
    top_k: int = TOP_K,
    video_id: str | None = None,
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
) -> list[tuple[Chunk, float, dict]]:
    """Hybrid retrieval combining BM25 and vector search.

    Args:
        query: User question
        top_k: Number of results to return
        video_id: Optional filter to specific video
        bm25_weight: Weight for BM25 scores (default 0.3)
        vector_weight: Weight for vector scores (default 0.7)

    Returns:
        List of (Chunk, combined_score, metadata_dict) tuples, sorted by combined score descending
        Metadata includes: 'vector_score', 'bm25_score', 'vector_rank', 'bm25_rank'
    """
    # Over-fetch from both retrievers to have candidates to combine
    over_fetch_k = max(top_k * 3, 20)

    # 1. Vector search (semantic)
    vector_results = vector_search(query, top_k=over_fetch_k, video_id=video_id, max_distance=2.0)

    # 2. BM25 search (lexical) over the entire corpus
    bm25_results = bm25_search(query, top_k=over_fetch_k, video_id=video_id)

    # 3. Normalize scores to [0, 1] range
    # Vector: distance is already in [0, 1] range where lower is better
    # Convert to similarity: similarity = 1 - distance
    vector_similarities: dict[str, float] = {}
    for chunk, distance in vector_results:
        vector_similarities[chunk.chunk_id] = 1.0 - distance

    # BM25: normalize by max score
    bm25_scores: dict[str, float] = {}
    for chunk, score in bm25_results:
        bm25_scores[chunk.chunk_id] = score

    max_vector = max(vector_similarities.values()) if vector_similarities else 1.0
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0

    vector_norm: dict[str, float] = {}
    if max_vector > 0:
        vector_norm = {cid: score / max_vector for cid, score in vector_similarities.items()}
    else:
        vector_norm = vector_similarities

    bm25_norm: dict[str, float] = {}
    if max_bm25 > 0:
        bm25_norm = {cid: score / max_bm25 for cid, score in bm25_scores.items()}
    else:
        bm25_norm = bm25_scores

    # 4. Gather union of candidate chunks
    all_chunk_ids = set(vector_norm.keys()) | set(bm25_norm.keys())

    # 5. Combine scores (weighted average)
    combined_scores: dict[str, tuple[Chunk, float, dict]] = {}
    for chunk_id in all_chunk_ids:
        # Get the chunk object from either result set
        chunk = None
        for c, _ in vector_results:
            if c.chunk_id == chunk_id:
                chunk = c
                break
        if chunk is None:
            for c, _ in bm25_results:
                if c.chunk_id == chunk_id:
                    chunk = c
                    break

        # If still None, skip (should not happen)
        if chunk is None:
            continue

        v_norm = vector_norm.get(chunk_id, 0.0)
        b_norm = bm25_norm.get(chunk_id, 0.0)

        # Weighted combination
        combined = (vector_weight * v_norm) + (bm25_weight * b_norm)

        metadata = {
            'vector_score': round(v_norm, 4),
            'bm25_score': round(b_norm, 4),
            'vector_rank': None,  # Will fill after sorting
            'bm25_rank': None,
        }
        combined_scores[chunk_id] = (chunk, combined, metadata)

    # Sort by combined score (descending)
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x[1],
        reverse=True
    )

    # Add rank information
    for rank, (chunk, score, metadata) in enumerate(sorted_results, 1):
        # Find vector rank
        for v_rank, (v_chunk, _) in enumerate(vector_results, 1):
            if v_chunk.chunk_id == chunk.chunk_id:
                metadata['vector_rank'] = v_rank
                break
        # Find BM25 rank
        sorted_by_bm25 = sorted(
            bm25_results,
            key=lambda x: x[1],
            reverse=True
        )
        for b_rank, (b_chunk, _) in enumerate(sorted_by_bm25, 1):
            if b_chunk.chunk_id == chunk.chunk_id:
                metadata['bm25_rank'] = b_rank
                break

    return sorted_results[:top_k]
