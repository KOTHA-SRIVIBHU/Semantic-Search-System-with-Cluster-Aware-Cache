import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt
import config

def get_representative_docs(probs, texts, n=5):
    """For each cluster, return indices of top‑n documents by probability."""
    n_clusters = probs.shape[1]
    representatives = {}
    for c in range(n_clusters):
        top_indices = np.argsort(probs[:, c])[-n:][::-1]
        representatives[c] = [(idx, probs[idx, c]) for idx in top_indices]
    return representatives

def get_cluster_keywords(texts, probs, n_keywords=10):
    """Extract keywords per cluster using TF‑IDF weighted by cluster probabilities."""
    n_clusters = probs.shape[1]
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    keywords = {}
    for c in range(n_clusters):
        # Weight each document's TF‑IDF by its probability for this cluster
        weighted = X.multiply(probs[:, c].reshape(-1, 1))
        avg_weights = np.asarray(weighted.mean(axis=0)).flatten()
        top_indices = np.argsort(avg_weights)[-n_keywords:][::-1]
        keywords[c] = [feature_names[i] for i in top_indices]
    return keywords

def visualize_clusters(embeddings, probs, save_path):
    """UMAP projection colored by dominant cluster."""
    reducer = umap.UMAP(random_state=42)
    emb_2d = reducer.fit_transform(embeddings)
    dominant = np.argmax(probs, axis=1)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=dominant, cmap='tab20', s=5, alpha=0.7)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('Document Clusters (UMAP projection)')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Cluster visualization saved to {save_path}")

if __name__ == "__main__":
    with open(config.TEXTS_FILE, "rb") as f:
        texts = pickle.load(f)
    embeddings = np.load(config.EMBEDDINGS_FILE)
    probs = np.load(config.CLUSTER_PROBS_FILE)

    reps = get_representative_docs(probs, texts)
    print("Representative documents per cluster:")
    for c, docs in reps.items():
        print(f"Cluster {c}: indices {[idx for idx, _ in docs]}")

    keywords = get_cluster_keywords(texts, probs)
    print("\nKeywords per cluster:")
    for c, kw in keywords.items():
        print(f"Cluster {c}: {', '.join(kw[:5])}")

    visualize_clusters(embeddings, probs, config.UMAP_PLOT_FILE)