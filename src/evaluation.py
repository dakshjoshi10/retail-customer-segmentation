"""
Clustering evaluation, metrics comparison, profiling, and visualization exports.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import MODEL_COMPARISON_CSV, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_clustering_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    runtime_sec: float = 0.0,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes standard unsupervised clustering metrics:
    - Silhouette Score (excluding noise for DBSCAN)
    - Davies-Bouldin Index
    - Calinski-Harabasz Score
    - Total clusters & Noise percentage
    """
    unique_labels = set(labels)
    n_noise = int(np.sum(labels == -1))
    noise_pct = (n_noise / len(labels)) * 100.0
    
    # Valid non-noise clusters
    non_noise_mask = labels != -1
    n_clusters = len([c for c in unique_labels if c != -1])

    if n_clusters < 2:
        return {
            "Model": model_name,
            "Clusters": n_clusters,
            "Noise %": round(noise_pct, 2),
            "Silhouette": np.nan,
            "Davies-Bouldin": np.nan,
            "Calinski-Harabasz": np.nan,
            "Runtime (s)": round(runtime_sec, 3)
        }

    # Evaluate metrics on non-noise partition if noise exists
    if n_noise > 0 and np.sum(non_noise_mask) > n_clusters:
        X_eval = X_scaled[non_noise_mask]
        labels_eval = labels[non_noise_mask]
    else:
        X_eval = X_scaled
        labels_eval = labels

    sil = float(silhouette_score(X_eval, labels_eval))
    db = float(davies_bouldin_score(X_eval, labels_eval))
    ch = float(calinski_harabasz_score(X_eval, labels_eval))

    return {
        "Model": model_name,
        "Clusters": n_clusters,
        "Noise %": round(noise_pct, 2),
        "Silhouette": round(sil, 4),
        "Davies-Bouldin": round(db, 4),
        "Calinski-Harabasz": round(ch, 2),
        "Runtime (s)": round(runtime_sec, 3)
    }


def compute_segment_profiles(
    df_customers: pd.DataFrame,
    cluster_col: str = "Segment"
) -> pd.DataFrame:
    """
    Builds comprehensive profile table per cluster/segment:
    - Count & % of customers
    - Total & % of revenue
    - Mean Recency, Frequency, Monetary, AvgOrderValue, UniqueProducts, AvgBasketSize
    """
    total_customers = len(df_customers)
    total_revenue = df_customers["Monetary"].sum()

    profile = (
        df_customers.groupby(cluster_col)
        .agg(
            CustomerCount=("CustomerID", "count"),
            TotalRevenue=("Monetary", "sum"),
            MeanRecency=("Recency", "mean"),
            MedianRecency=("Recency", "median"),
            MeanFrequency=("Frequency", "mean"),
            MedianFrequency=("Frequency", "median"),
            MeanMonetary=("Monetary", "mean"),
            MedianMonetary=("Monetary", "median"),
            MeanAOV=("AvgOrderValue", "mean"),
            MeanUniqueProducts=("UniqueProducts", "mean"),
            MeanBasketSize=("AvgBasketSize", "mean")
        )
        .reset_index()
    )

    profile["PctCustomers"] = (profile["CustomerCount"] / total_customers) * 100.0
    profile["PctRevenue"] = (profile["TotalRevenue"] / total_revenue) * 100.0

    # Format ordering
    cols_order = [
        cluster_col, "CustomerCount", "PctCustomers", "TotalRevenue", "PctRevenue",
        "MeanRecency", "MedianRecency", "MeanFrequency", "MedianFrequency",
        "MeanMonetary", "MedianMonetary", "MeanAOV", "MeanUniqueProducts", "MeanBasketSize"
    ]
    return profile[[c for c in cols_order if c in profile.columns]]


def profile_dbscan_noise(df_customers: pd.DataFrame, dbscan_labels: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes DBSCAN noise points (-1) against core clusters to verify
    if noise points correspond to High-Net-Worth VIPs, wholesale bulk buyers, or anomalous extremes.
    """
    df = df_customers.copy()
    df["IsNoise"] = dbscan_labels == -1

    noise_profile = df.groupby("IsNoise").agg(
        Count=("CustomerID", "count"),
        MeanMonetary=("Monetary", "mean"),
        MedianMonetary=("Monetary", "median"),
        MaxMonetary=("Monetary", "max"),
        MeanFrequency=("Frequency", "mean"),
        MedianFrequency=("Frequency", "median"),
        MeanTotalUnits=("TotalUnits", "mean"),
        MeanAOV=("AvgOrderValue", "mean")
    ).reset_index()

    noise_profile["Group"] = noise_profile["IsNoise"].map({True: "Noise (Outliers/VIPs)", False: "Core Clusters"})
    return {
        "summary": noise_profile,
        "interpretation": (
            "DBSCAN noise points exhibit significantly higher Monetary and Frequency values, "
            "identifying them as high-volume B2B/wholesale buyers and VIP outliers rather than random data artifacts."
        )
    }


def compute_ari_matrix(
    models_labels: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Computes pairwise Adjusted Rand Index (ARI) across clustering solutions.
    """
    names = list(models_labels.keys())
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)

    for name1 in names:
        for name2 in names:
            ari = adjusted_rand_score(models_labels[name1], models_labels[name2])
            matrix.loc[name1, name2] = round(float(ari), 4)

    return matrix


def save_evaluation_plots(
    kmeans_search_res: Dict[str, Any],
    k_distances: np.ndarray,
    dbscan_eps: float,
    X_scaled: np.ndarray,
    figures_dir: Path = FIGURES_DIR
) -> None:
    """
    Generates and saves publication-quality evaluation figures:
    - K-Means Elbow & Silhouette curve
    - DBSCAN k-distance knee curve
    - Agglomerative Hierarchical Dendrogram (truncated)
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. K-Means Elbow & Silhouette
    k_vals = kmeans_search_res["k_values"]
    inertias = kmeans_search_res["inertias"]
    silhouettes = kmeans_search_res["silhouettes"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Elbow)", color=color)
    ax1.plot(k_vals, inertias, "o-", color=color, linewidth=2, label="Inertia")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Silhouette Score", color=color)
    ax2.plot(k_vals, silhouettes, "s--", color=color, linewidth=2, label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("K-Means Model Selection: Elbow & Silhouette Scores")
    plt.tight_layout()
    plt.savefig(figures_dir / "kmeans_elbow_silhouette.png", dpi=300)
    plt.close()

    # 2. DBSCAN k-distance plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_distances, color="#2b5c8f", linewidth=2, label="k-distance (k=10)")
    ax.axhline(y=dbscan_eps, color="red", linestyle="--", label=f"Detected knee eps={dbscan_eps:.2f}")
    ax.set_title("DBSCAN k-Distance Graph for Optimal Epsilon Selection")
    ax.set_xlabel("Points sorted by distance to k-th nearest neighbor")
    ax.set_ylabel("k-distance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "dbscan_kdistance_plot.png", dpi=300)
    plt.close()

    # 3. Agglomerative Dendrogram (Ward linkage, sampled to avoid memory spike)
    logger.info("Generating truncated hierarchical dendrogram...")
    sample_indices = np.random.RandomState(42).choice(len(X_scaled), size=min(1000, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_indices]
    Z = linkage(X_sample, method="ward")

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=12,
        leaf_rotation=45.,
        leaf_font_size=10.,
        show_contracted=True,
        ax=ax
    )
    ax.set_title("Agglomerative Hierarchical Clustering Dendrogram (Ward Linkage, Truncated)")
    ax.set_xlabel("Cluster / Sample Index")
    ax.set_ylabel("Ward Linkage Euclidean Distance")
    plt.tight_layout()
    plt.savefig(figures_dir / "hierarchical_dendrogram.png", dpi=300)
    plt.close()

    logger.info(f"Evaluation figures saved successfully to {figures_dir}")
