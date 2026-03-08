# main.py
from fastapi import FastAPI
import pickle
import numpy as np
from contextlib import asynccontextmanager
import config
from preprocessing.data_loader import load_and_clean
from embeddings.embedder import embed_texts
from vector_store.faiss_index import FaissIndex
from clustering.gmm_cluster import fit_gmm
from retrieval.searcher import Searcher
from cache.cluster_cache import ClusterCacheManager
import api.routes  # import the whole module to set its globals

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything on startup
    print("Loading data and models...")

    # Texts
    with open(config.TEXTS_FILE, "rb") as f:
        texts = pickle.load(f)

    # Embeddings (or recompute if missing)
    try:
        embeddings = np.load(config.EMBEDDINGS_FILE)
    except FileNotFoundError:
        print("Embeddings not found, computing now...")
        embeddings = embed_texts(texts)
        np.save(config.EMBEDDINGS_FILE, embeddings)

    # FAISS index
    index = FaissIndex()
    try:
        index.load(config.FAISS_INDEX_FILE)
    except:
        print("FAISS index not found, building now...")
        index.build(embeddings)
        index.save(config.FAISS_INDEX_FILE)

    # Searcher
    api.routes.searcher = Searcher(index, texts)

    # GMM model
    try:
        with open(config.GMM_MODEL_FILE, "rb") as f:
            gmm_model = pickle.load(f)
    except:
        print("GMM model not found, fitting now...")
        gmm_model, probs, _ = fit_gmm(embeddings)
        with open(config.GMM_MODEL_FILE, "wb") as f:
            pickle.dump(gmm_model, f)
        np.save(config.CLUSTER_PROBS_FILE, probs)
    api.routes.gmm_model = gmm_model

    # Cache manager
    api.routes.cache_manager = ClusterCacheManager(n_clusters=gmm_model.n_components)

    print("Startup complete.")
    yield
    # Shutdown: optional persistence of cache? Not required.

app = FastAPI(lifespan=lifespan)
app.include_router(api.routes.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)