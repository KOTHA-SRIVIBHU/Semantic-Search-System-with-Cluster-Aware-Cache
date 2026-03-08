import faiss
import numpy as np
import config

class FaissIndex:
    def __init__(self):
        self.index = None
        self.embeddings = None

    def build(self, embeddings: np.ndarray):
        """Build a FAISS index with inner product (cosine on normalized vectors)."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product = cosine on unit vectors
        self.index.add(embeddings)
        self.embeddings = embeddings
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def search(self, query_emb: np.ndarray, k: int = config.TOP_K_RESULTS):
        """Return (scores, indices) for top‑k similar documents."""
        if self.index is None:
            raise ValueError("Index not built/loaded.")
        scores, indices = self.index.search(query_emb.reshape(1, -1), k)
        return scores[0], indices[0]

if __name__ == "__main__":
    embeddings = np.load(config.EMBEDDINGS_FILE)
    index = FaissIndex()
    index.build(embeddings)
    index.save(config.FAISS_INDEX_FILE)
    print(f"FAISS index saved to {config.FAISS_INDEX_FILE}")