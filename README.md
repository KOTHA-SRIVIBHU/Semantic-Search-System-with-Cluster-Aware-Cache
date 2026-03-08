# Semantic Search with Cluster‑Aware Caching

This project implements a semantic search system over the 20 Newsgroups dataset. It features:
- Fuzzy clustering (GMM) to capture overlapping topics.
- A custom, cluster‑partitioned semantic cache with LRU eviction.
- FastAPI endpoints for querying and cache introspection.

## Architecture

1. **Data Preparation**: Load and clean the 20 Newsgroups corpus (headers/footers/quotes removed).
2. **Embeddings**: `all-MiniLM-L6-v2` from sentence‑transformers (384‑d, normalized).
3. **Vector Store**: FAISS index with inner product (cosine) for fast retrieval.
4. **Fuzzy Clustering**: Gaussian Mixture Model with BIC‑based component selection. Each document gets a probability distribution over clusters.
5. **Semantic Cache**: 
   - A `ClusterCacheManager` holds one `ClusterCache` per cluster.
   - Lookup: embed query → predict dominant cluster → search only that cluster’s cache via cosine similarity.
   - LRU eviction and optional TTL per cache.
   - Cache hit/miss stats are aggregated globally.
6. **Retrieval**: On cache miss, search FAISS for top‑k similar documents.
7. **API**:
   - `POST /query` – returns results with cache hit info.
   - `GET /cache/stats` – overall cache statistics.
   - `DELETE /cache` – flush all caches.

## Why This Design?

- **Embedding model**: `all-MiniLM-L6-v2` balances speed and quality; 384‑d is sufficient for clustering and caching.
- **Fuzzy clustering**: GMM provides soft assignments, matching the real overlapping nature of news topics.
- **Cluster‑partitioned cache**: Reduces search space (only one cluster’s cache per query) and allows per‑cluster tuning.
- **Custom cache**: No external dependencies; full control over eviction and similarity logic.

## Cluster Analysis

- Number of clusters determined by BIC (typically 18–24 for this dataset).
- Representative documents and keywords extracted for each cluster (see `cluster_analysis.py`).
- UMAP visualization saved as `data/cluster_visualization.png`.

## Cache Threshold Tuning

The similarity threshold (default 0.85) controls cache aggressiveness:
- **Low threshold** → more hits, but risk of returning semantically different results.
- **High threshold** → fewer hits, higher precision.
The value can be changed in `config.py` (CACHE_THRESHOLD). You can experiment to find the best balance for your use case.

## How to Run

### Using Python (local)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Precompute data (embeddings, clustering, etc.)
python scripts/precompute.py

# 4. Start the API
uvicorn main:app --reload