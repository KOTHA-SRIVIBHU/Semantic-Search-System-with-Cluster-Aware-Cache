import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from typing import List
import config

_model = None

LOCAL_MODEL_FILES = ("config.json", "modules.json")
REMOTE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def resolve_model_source() -> str:
    model_path = config.EMBEDDING_MODEL

    if os.path.isdir(model_path):
        missing_files = [
            filename
            for filename in LOCAL_MODEL_FILES
            if not os.path.exists(os.path.join(model_path, filename))
        ]
        if not missing_files:
            return model_path

        print(
            f"Local model directory '{model_path}' is incomplete "
            f"(missing: {', '.join(missing_files)}). "
            f"Falling back to {REMOTE_MODEL_ID}."
        )

    return REMOTE_MODEL_ID

def get_embedder():
    global _model
    if _model is None:
        model_source = resolve_model_source()
        print(f"Loading embedding model: {model_source}")
        _model = SentenceTransformer(model_source)
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
    config.ensure_data_dir()
    with open(config.TEXTS_FILE, "rb") as f:
        texts = pickle.load(f)
    embeddings = embed_texts(texts)
    np.save(config.EMBEDDINGS_FILE, embeddings)
    print(f"Saved embeddings to {config.EMBEDDINGS_FILE}")
