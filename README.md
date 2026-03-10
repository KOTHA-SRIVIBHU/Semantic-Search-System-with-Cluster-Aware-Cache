# Semantic Search with Cluster-Aware Caching

This project implements a semantic search system over the **20 Newsgroups dataset**. It features:

- **Fuzzy clustering** (Gaussian Mixture Model) to capture overlapping topics.
- **Custom cluster-partitioned semantic cache** with LRU eviction, built from scratch.
- **FastAPI** service with three endpoints.
- **FAISS** vector index for efficient similarity search.
- **UMAP** visualisation of document clusters.

---

# System Architecture

![Architecture Diagram](docs/architecture.png)

1. **Data Preparation** – Load 20 Newsgroups, remove headers/footers/quotes, clean text.  
2. **Embeddings** – `all-MiniLM-L6-v2` (384-dim, normalized) via Sentence-Transformers.  
3. **Fuzzy Clustering** – GMM with BIC-based component selection (30 clusters).  
4. **Vector Store** – FAISS index (inner product) for cosine similarity search.  
5. **Semantic Cache** – `ClusterCacheManager` with per-cluster `ClusterCache` (LRU, TTL optional).  
6. **API** – FastAPI with `/query`, `/cache/stats`, and `DELETE /cache`.

---

# Requirements

- Python **3.8 – 3.10** (tested on **3.10**)  
- See `requirements.txt` for package versions.

---

# Setup & Installation

## 1. Clone the repository

```bash
git clone https://github.com/yourusername/trademarkia-search.git
cd trademarkia-search
```

## 2. Create and activate a virtual environment

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Download the embedding model locally (to avoid timeouts)

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir ./models/all-MiniLM-L6-v2 \
  --include "pytorch_model.bin" \
  --include "*.json" \
  --include "vocab.txt" \
  --include "modules.json" \
  --include "special_tokens_map.json"
```

If `huggingface-cli` is not available:

```bash
pip install huggingface-hub
```

---

## 5. Run the precomputation script

This loads the dataset, generates embeddings, builds the FAISS index, and fits the GMM.

```bash
python -m scripts.precompute
```

This will create the following files in the **data/** directory:

```
texts.pkl
embeddings.npy
faiss.index
gmm.pkl
cluster_probs.npy
centroids.npy
cluster_visualization.png
```

**Note**

The first run downloads the **20 Newsgroups dataset** and may take **10–20 minutes on CPU**.

---

# Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

API will run at:

```
http://localhost:8000
```

Interactive documentation:

```
http://localhost:8000/docs
```

---

# API Endpoints

| Method | Endpoint | Description |
|------|------|------|
| POST | `/query` | Submit a query and return search results |
| GET | `/cache/stats` | View cache statistics |
| DELETE | `/cache` | Flush entire cache |

---

# Example Request

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Space shuttle launch"}'
```

---

# Example Response (Cache Miss)

```json
{
  "query": "Space shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "dominant_cluster": 18,
  "results": [
    {
      "document_id": 4672,
      "text": "Archive-name: space/schedule ...",
      "similarity": 0.6015
    }
  ]
}
```

---

# Cache Design

The cache is **cluster-partitioned**.

- A `ClusterCacheManager` holds one **ClusterCache per cluster**.
- Each cluster cache uses an **OrderedDict** to implement **LRU eviction**.

### Query Flow

1. Embed the query  
2. Predict dominant cluster using GMM  
3. Search **only that cluster cache** for similar queries  
4. If similarity ≥ threshold → **Cache Hit**  
5. Otherwise → search FAISS and store result in cache

The similarity threshold (**default 0.85**) is configurable in `config.py`.

---

# Cluster Analysis

The GMM selected **30 clusters** using **BIC**.

| Cluster | Top Keywords |
|------|------|
| 0 | drive, scsi, disk, drives, ide, hard |
| 6 | god, religion, believe, atheism |
| 7 | baseball, game, runs, players |
| 11 | armenian, turkey, genocide |
| 15 | car, engine, miles |
| 18 | space, nasa, launch, mission |
| 21 | windows, dos, files |
| 24 | jesus, bible, church |
| 26 | government, rights, law |
| 29 | gun, weapons, crime |

A **UMAP visualization** is saved as:

```
data/cluster_visualization.png
```

---

# Docker

A **Dockerfile** is provided.

## Build Image

```bash
docker build -t trademarkia-search .
```

## Run Container

```bash
docker run -p 8000:8000 trademarkia-search
```

The Docker build runs the **precomputation script**, so models and data are included in the image.

---

# Project Structure

```
.
├── api/
├── cache/
├── clustering/
├── config.py
├── data/
├── embeddings/
├── main.py
├── models/
├── preprocessing/
├── retrieval/
├── scripts/
├── utils/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Tuning the Cache Threshold

The key parameter is:

```
CACHE_THRESHOLD
```

Location:

```
config.py
```

Typical values:

```
0.8 – 0.9
```

Lower threshold → more hits but less precise  
Higher threshold → fewer hits but more precise

---

# Notes

- `data/` and `models/` are **ignored in git**.
- They are generated by the **precomputation script**.
- If embedding runs out of memory, reduce `batch_size` in:

```
embeddings/embedder.py
```

- The system is **thread-safe** (cache uses locks).

---