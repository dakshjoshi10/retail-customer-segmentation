"""
Clustering models implementation: K-Means, DBSCAN, and Agglomerative Hierarchical Clustering.
"""

import logging
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from src.config import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_kmeans_search(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11)
) -> Dict[str, Any]:
    """
    Evaluates K-Means clustering across a range of k values (2..10) using:
    - Inertia (Sum of squared distances)
    - Silhouette score
    - Calinski-Harabasz score
    - Davies-Bouldin index
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    inertias = []
    silhouettes = []
    ch_scores = []
    db_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, labels)))
        ch_scores.append(float(calinski_harabasz_score(X_scaled, labels)))
        db_scores.append(float(davies_bouldin_score(X_scaled, labels)))

    # Knee locator on inertia curve
    k_list = list(k_range)
    knee_locator = KneeLocator(k_list, inertias, curve="convex", direction="decreasing")
    optimal_k = knee_locator.elbow if knee_locator.elbow is not None else 4

    return {
        "k_values": k_list,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "calinski_harabasz": ch_scores,
        "davies_bouldin": db_scores,
        "elbow_k": optimal_k
    }


def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 4
) -> Tuple[np.ndarray, KMeans, float]:
    """
    Fits K-Means with specified number of clusters and fixed seed.
    """
    start_time = time.time()
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    runtime = time.time() - start_time
    return labels, km, runtime


def run_dbscan_kdistance(
    X_scaled: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, float]:
    """
    Computes k-distance plot array and detects knee/elbow for eps selection.
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, k - 1])
    
    # Locate knee on the sorted k-distances curve
    x_indices = np.arange(len(k_distances))
    knee_loc = KneeLocator(x_indices, k_distances, curve="convex", direction="increasing")
    detected_eps = float(k_distances[knee_loc.elbow]) if knee_loc.elbow is not None else 0.8
    return k_distances, detected_eps


def grid_search_dbscan(
    X_scaled: np.ndarray,
    eps_values: list = None,
    min_samples_values: list = None
) -> Dict[str, Any]:
    """
    Grid searches DBSCAN hyperparameters (eps, min_samples) to find optimal clustering:
    Prioritizes models finding >= 2 distinct clusters with reasonable noise ratio (< 15%).
    """
    from sklearn.metrics import silhouette_score
    if eps_values is None:
        eps_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if min_samples_values is None:
        min_samples_values = [5, 10, 15]

    best_score = -1.0
    best_params = {"eps": 0.6, "min_samples": 10}
    best_labels = None
    all_results = []

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = (np.sum(labels == -1) / len(labels)) * 100.0

            sil = np.nan
            if n_clusters >= 2 and np.sum(labels != -1) > n_clusters:
                sil = float(silhouette_score(X_scaled[labels != -1], labels[labels != -1]))
                if sil > best_score and noise_pct < 20.0:
                    best_score = sil
                    best_params = {"eps": eps, "min_samples": ms}
                    best_labels = labels

            all_results.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "noise_pct": round(noise_pct, 2),
                "silhouette": round(sil, 4) if not np.isnan(sil) else np.nan
            })

    if best_labels is None:
        # Fallback to default
        db = DBSCAN(eps=0.6, min_samples=10)
        best_labels = db.fit_predict(X_scaled)
        best_params = {"eps": 0.6, "min_samples": 10}

    return {
        "best_params": best_params,
        "best_labels": best_labels,
        "best_silhouette": best_score,
        "grid_results": all_results
    }


def fit_dbscan(
    X_scaled: np.ndarray,
    eps: float = 0.6,
    min_samples: int = 10
) -> Tuple[np.ndarray, DBSCAN, float]:
    """
    Fits DBSCAN clustering.
    """
    start_time = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    runtime = time.time() - start_time
    return labels, db, runtime


def fit_agglomerative(
    X_scaled: np.ndarray,
    n_clusters: int = 4,
    linkage: str = "ward"
) -> Tuple[np.ndarray, AgglomerativeClustering, float]:
    """
    Fits Agglomerative Hierarchical Clustering.
    """
    start_time = time.time()
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X_scaled)
    runtime = time.time() - start_time
    return labels, agg, runtime


def compute_pca_projection(X_scaled: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Projects scaled feature space to 2D using Principal Component Analysis (PCA).
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_} (Total: {pca.explained_variance_ratio_.sum():.2%})")
    return X_pca


def assign_kmeans_segment_names(
    df_customers: pd.DataFrame,
    cluster_labels: np.ndarray
) -> pd.DataFrame:
    """
    Names K-Means segments based directly on empirical centroid statistics:
    - Monetary (Spend)
    - Frequency (Orders)
    - Recency (Days since last order)
    
    Rules derived systematically:
    - Highest Monetary + High Frequency + Low Recency -> 'Champions'
    - High-to-Moderate Monetary + High Frequency + Low Recency -> 'Loyal'
    - Moderate Monetary + Low/Moderate Frequency + Low Recency -> 'Potential Loyalists'
    - High/Moderate Spend/Freq in the past + High Recency -> 'At-Risk'
    - Lowest Monetary + Lowest Frequency + Highest Recency -> 'Lost / Hibernating'
    """
    df = df_customers.copy()
    df["Cluster"] = cluster_labels

    cluster_summary = df.groupby("Cluster").agg(
        MeanRecency=("Recency", "mean"),
        MeanFrequency=("Frequency", "mean"),
        MeanMonetary=("Monetary", "mean"),
        CustomerCount=("CustomerID", "count")
    ).reset_index()

    # Rank clusters based on RFM score composite
    # Higher Monetary and Frequency is positive; Higher Recency is negative
    r_rank = cluster_summary["MeanRecency"].rank(ascending=False) # low recency = high score
    f_rank = cluster_summary["MeanFrequency"].rank(ascending=True)
    m_rank = cluster_summary["MeanMonetary"].rank(ascending=True)

    cluster_summary["RFM_Score"] = r_rank + f_rank + m_rank
    sorted_clusters = cluster_summary.sort_values("RFM_Score", ascending=False)["Cluster"].tolist()

    num_clusters = len(sorted_clusters)
    name_map = {}

    if num_clusters == 4:
        # Standard 4-tier naming
        name_map[sorted_clusters[0]] = "Champions"
        name_map[sorted_clusters[1]] = "Loyal"
        name_map[sorted_clusters[2]] = "At-Risk"
        name_map[sorted_clusters[3]] = "Lost / Hibernating"
    elif num_clusters == 5:
        name_map[sorted_clusters[0]] = "Champions"
        name_map[sorted_clusters[1]] = "Loyal"
        name_map[sorted_clusters[2]] = "Potential Loyalists"
        name_map[sorted_clusters[3]] = "At-Risk"
        name_map[sorted_clusters[4]] = "Lost / Hibernating"
    else:
        # Generic tier fallback
        labels_pool = ["Champions", "Loyal", "Potential Loyalists", "Promising", "Needs Attention", "At-Risk", "Lost / Hibernating"]
        for idx, c in enumerate(sorted_clusters):
            name_map[c] = labels_pool[min(idx, len(labels_pool) - 1)]

    df["Segment"] = df["Cluster"].map(name_map)
    return df
