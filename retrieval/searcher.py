import numpy as np
import pickle
from vector_store.faiss_index import FaissIndex
import config

class Searcher:
    def __init__(self, index: FaissIndex, texts: list):
        self.index = index
        self.texts = texts

    def search(self, query_emb: np.ndarray, k: int = config.TOP_K_RESULTS):
        scores, indices = self.index.search(query_emb, k)
        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            results.append({
                'document_id': int(idx),
                'text': self.texts[idx],
                'similarity': float(score)   # cosine (since normalized)
            })
        return results