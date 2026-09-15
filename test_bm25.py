import time
from ytrag.retrieval import bm25_search
from ytrag.index import get_all_chunks

t0 = time.time()
res = bm25_search("dynamic programming maximum subarray sum", top_k=5)
t1 = time.time()
print(f"BM25 search took {t1-t0:.3f}s, found {len(res)} items")

if res:
    chunk, score = res[0]
    print(f"Chunk ID: {chunk.chunk_id}, Score: {score:.4f}")
