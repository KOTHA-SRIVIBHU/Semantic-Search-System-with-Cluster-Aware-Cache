from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from typing import List
import config

_model = None

def get_embedder():
    global _model
    if _model is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model

def embed_texts(texts: List[str], batch_size=64) -> np.ndarray:
    model = get_embedder()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def embed_query(query: str) -> np.ndarray:
    model = get_embedder()
    emb = model.encode([query])[0]
    return emb / np.linalg.norm(emb)

if __name__ == "__main__":
    with open(config.TEXTS_FILE, "rb") as f:
        texts = pickle.load(f)
    embeddings = embed_texts(texts)
    np.save(config.EMBEDDINGS_FILE, embeddings)
    print(f"Saved embeddings to {config.EMBEDDINGS_FILE}")