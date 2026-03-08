# scripts/precompute.py
from preprocessing.data_loader import load_and_clean
from embeddings.embedder import embed_texts
from vector_store.faiss_index import FaissIndex
from clustering.gmm_cluster import fit_gmm
from clustering.cluster_analysis import get_representative_docs, get_cluster_keywords, visualize_clusters
import config
import pickle
import numpy as np

def main():
    # 1. Load & clean
    texts = load_and_clean()
    with open(config.TEXTS_FILE, "wb") as f:
        pickle.dump(texts, f)

    # 2. Embed
    embeddings = embed_texts(texts)
    np.save(config.EMBEDDINGS_FILE, embeddings)

    # 3. Build FAISS index
    index = FaissIndex()
    index.build(embeddings)
    index.save(config.FAISS_INDEX_FILE)

    # 4. Fuzzy clustering
    gmm, probs, centroids = fit_gmm(embeddings)
    with open(config.GMM_MODEL_FILE, "wb") as f:
        pickle.dump(gmm, f)
    np.save(config.CLUSTER_PROBS_FILE, probs)
    np.save(config.CENTROIDS_FILE, centroids)

    # 5. Analysis & visualization
    reps = get_representative_docs(probs, texts)
    print("Representative documents:", reps)
    keywords = get_cluster_keywords(texts, probs)
    print("Keywords:", keywords)
    visualize_clusters(embeddings, probs, config.UMAP_PLOT_FILE)

if __name__ == "__main__":
    main()