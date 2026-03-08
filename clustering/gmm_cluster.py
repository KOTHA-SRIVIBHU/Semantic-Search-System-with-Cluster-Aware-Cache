import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
from sklearn.metrics import silhouette_score
import config

def select_n_clusters(embeddings, min_k=10, max_k=30, step=2):
    """Select best number of clusters using BIC."""
    bic_scores = []
    k_range = range(min_k, max_k+1, step)
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='diag')
        gmm.fit(embeddings)
        bic_scores.append(gmm.bic(embeddings))
        print(f"k={k}, BIC={bic_scores[-1]:.2f}")
    best_k = k_range[np.argmin(bic_scores)]
    print(f"Selected k={best_k} based on lowest BIC.")
    return best_k

def fit_gmm(embeddings, n_clusters=None):
    if n_clusters is None:
        n_clusters = select_n_clusters(embeddings, config.MIN_CLUSTERS, config.MAX_CLUSTERS, config.BIC_STEP)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='diag')
    gmm.fit(embeddings)
    probs = gmm.predict_proba(embeddings)          # fuzzy memberships
    centroids = gmm.means_                          # cluster centres
    return gmm, probs, centroids

if __name__ == "__main__":
    embeddings = np.load(config.EMBEDDINGS_FILE)
    gmm, probs, centroids = fit_gmm(embeddings)
    with open(config.GMM_MODEL_FILE, "wb") as f:
        pickle.dump(gmm, f)
    np.save(config.CLUSTER_PROBS_FILE, probs)
    np.save(config.CENTROIDS_FILE, centroids)
    print(f"GMM model saved, clusters: {gmm.n_components}")