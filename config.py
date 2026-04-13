import os

# Paths
DATA_DIR = "data"
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.npy"
TEXTS_FILE = f"{DATA_DIR}/texts.pkl"
FAISS_INDEX_FILE = f"{DATA_DIR}/faiss.index"
GMM_MODEL_FILE = f"{DATA_DIR}/gmm.pkl"
CLUSTER_PROBS_FILE = f"{DATA_DIR}/cluster_probs.npy"
CENTROIDS_FILE = f"{DATA_DIR}/centroids.npy"
UMAP_PLOT_FILE = f"{DATA_DIR}/cluster_visualization.png"


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

# Embeddings
EMBEDDING_MODEL = "./models/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Clustering
MIN_CLUSTERS = 10
MAX_CLUSTERS = 30
BIC_STEP = 2          # evaluate every 2 clusters for speed

# Cache
CACHE_THRESHOLD = 0.85          # cosine similarity threshold
GLOBAL_CACHE_SIZE = 1000        # total entries across all clusters
PER_CLUSTER_CACHE_SIZE = 200    # max per cluster if not overridden
CACHE_TTL = 3600                 # seconds (optional, None to disable)

# Retrieval
TOP_K_RESULTS = 5

# API
HOST = "0.0.0.0"
PORT = 8000
