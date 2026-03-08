from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import config
from embeddings.embedder import embed_query
from retrieval.searcher import Searcher
from cache.cluster_cache import ClusterCacheManager
import numpy as np

router = APIRouter()

# These will be set by main app
searcher: Optional[Searcher] = None
cache_manager: Optional[ClusterCacheManager] = None
gmm_model = None   # for cluster prediction

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    dominant_cluster: int
    results: List[dict]

class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    query = request.query

    # 1. Embed
    q_emb = embed_query(query)

    # 2. Determine dominant cluster using GMM
    probs = gmm_model.predict_proba(q_emb.reshape(1, -1))[0]
    dominant_cluster = int(np.argmax(probs))

    # 3. Check cache
    cache_result = cache_manager.lookup(q_emb, dominant_cluster)
    if cache_result:
        matched_query, similarity, cached_results = cache_result
        return QueryResponse(
            query=query,
            cache_hit=True,
            matched_query=matched_query,
            similarity_score=similarity,
            dominant_cluster=dominant_cluster,
            results=cached_results
        )

    # 4. Cache miss: perform search
    results = searcher.search(q_emb)

    # 5. Store in cache
    cache_manager.insert(query, q_emb, dominant_cluster, results)

    return QueryResponse(
        query=query,
        cache_hit=False,
        dominant_cluster=dominant_cluster,
        results=results
    )

@router.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    return cache_manager.get_stats()

@router.delete("/cache")
async def flush_cache():
    cache_manager.flush()
    return {"message": "Cache flushed"}