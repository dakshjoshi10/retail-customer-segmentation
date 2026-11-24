"""
End-to-end execution pipeline for retail customer segmentation and sales intelligence.
Runs the complete flow: Load -> Clean -> Product Analysis -> Features -> Cluster -> Evaluate -> Extras -> Export.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    CUSTOMERS_SEGMENTED_CSV,
    MODEL_COMPARISON_CSV,
    TOP_PRODUCTS_CSV,
    PRODUCT_CLUSTERS_CSV,
    MARKET_BASKET_RULES_CSV,
    SEGMENT_PROFILES_CSV,
    FIGURES_DIR,
    DEFAULT_KMEANS_K
)
from src.data_loader import load_raw_data
from src.cleaning import clean_retail_data, get_customer_clustering_data
from src.product_analysis import analyze_products
from src.features import compute_customer_features, prepare_features_for_clustering
from src.clustering import (
    run_kmeans_search,
    fit_kmeans,
    run_dbscan_kdistance,
    grid_search_dbscan,
    fit_dbscan,
    fit_agglomerative,
    compute_pca_projection,
    assign_kmeans_segment_names
)
from src.evaluation import (
    compute_clustering_metrics,
    compute_segment_profiles,
    profile_dbscan_noise,
    compute_ari_matrix,
    save_evaluation_plots
)
from src.insights import (
    compute_segment_product_insights,
    cluster_products,
    run_market_basket_analysis,
    generate_business_recommendations
)
from src.export_deploy import export_deploy_assets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Pipeline")


def run_pipeline():
    logger.info("=" * 70)
    logger.info("STARTING RETAIL CUSTOMER SEGMENTATION & SALES INTELLIGENCE PIPELINE")
    logger.info("=" * 70)

    # 1. Load Raw Data
    logger.info("\n--- STEP 1: LOAD RAW DATA ---")
    df_raw = load_raw_data(RAW_DATA_FILE)

    # 2. Clean Data (with documented row counts)
    logger.info("\n--- STEP 2: DATA CLEANING ---")
    df_cleaned, counts_log = clean_retail_data(df_raw, save_processed=True)
    logger.info("Cleaning Row Counts Summary:")
    for step, count in counts_log.items():
        logger.info(f"  {step}: {count:,} rows")

    # 3. Product Sales Analysis
    logger.info("\n--- STEP 3: PRODUCT SALES & PERFORMANCE ANALYSIS ---")
    product_results = analyze_products(df_cleaned, save_csv=True)
    most_sold_units = product_results["most_sold_by_units"]
    top_rev = product_results["top_by_revenue"]
    top_ord = product_results["top_by_orders"]
    logger.info(f"1. Most Sold Product (Units): {most_sold_units['Description']} ({most_sold_units['TotalUnits']:,} units)")
    logger.info(f"2. Top Product by Revenue: {top_rev['Description']} (£{top_rev['TotalRevenue']:,.2f})")
    logger.info(f"3. Top Product by Orders: {top_ord['Description']} ({top_ord['DistinctOrders']:,} orders)")

    # 4. Customer Behavioral Feature Engineering
    logger.info("\n--- STEP 4: CUSTOMER FEATURE ENGINEERING ---")
    df_cust_raw = get_customer_clustering_data(df_cleaned)
    features_df = compute_customer_features(df_cust_raw)
    features_df, df_log, X_scaled, scaler = prepare_features_for_clustering(features_df, save_parquet=True)

    # 5. Clustering Algorithms Execution & Model Selection
    logger.info("\n--- STEP 5: CLUSTERING MODELS EXECUTION ---")
    
    # 5A. K-Means Search & Fit
    logger.info("Evaluating K-Means (k=2..10)...")
    km_search = run_kmeans_search(X_scaled)
    selected_k = km_search["elbow_k"]
    if selected_k < 3 or selected_k > 6:
        selected_k = DEFAULT_KMEANS_K
    logger.info(f"Selected K-Means k={selected_k} based on elbow inertia and silhouette curve.")
    km_labels, km_model, km_runtime = fit_kmeans(X_scaled, n_clusters=selected_k)
    km_metrics = compute_clustering_metrics(X_scaled, km_labels, runtime_sec=km_runtime, model_name=f"K-Means (k={selected_k})")

    # 5B. DBSCAN Epsilon Knee & Grid Search Fit
    logger.info("Evaluating DBSCAN k-distance knee and hyperparameter grid search...")
    k_distances, knee_eps = run_dbscan_kdistance(X_scaled, k=10)
    logger.info(f"Detected knee epsilon: {knee_eps:.3f}")
    
    # Grid search around knee for best eps and min_samples
    db_search = grid_search_dbscan(
        X_scaled,
        eps_values=[0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        min_samples_values=[5, 10, 15]
    )
    best_eps = db_search["best_params"]["eps"]
    best_ms = db_search["best_params"]["min_samples"]
    logger.info(f"Optimal DBSCAN parameters: eps={best_eps}, min_samples={best_ms} (Silhouette={db_search['best_silhouette']:.4f})")
    
    db_labels, db_model, db_runtime = fit_dbscan(X_scaled, eps=best_eps, min_samples=best_ms)
    db_metrics = compute_clustering_metrics(X_scaled, db_labels, runtime_sec=db_runtime, model_name=f"DBSCAN (eps={best_eps:.2f}, ms={best_ms})")

    # 5C. Agglomerative Hierarchical Clustering
    logger.info("Fitting Agglomerative Clustering (Ward Linkage)...")
    agg_labels, agg_model, agg_runtime = fit_agglomerative(X_scaled, n_clusters=selected_k, linkage="ward")
    agg_metrics = compute_clustering_metrics(X_scaled, agg_labels, runtime_sec=agg_runtime, model_name=f"Agglomerative (Ward, k={selected_k})")

    # 5D. Compare Linkages for Agglomerative
    for link in ["complete", "average"]:
        link_labels, _, link_runtime = fit_agglomerative(X_scaled, n_clusters=selected_k, linkage=link)
        link_metrics = compute_clustering_metrics(X_scaled, link_labels, runtime_sec=link_runtime, model_name=f"Agglomerative ({link.capitalize()}, k={selected_k})")
        logger.info(f"Linkage {link}: Silhouette = {link_metrics['Silhouette']}, Calinski-Harabasz = {link_metrics['Calinski-Harabasz']}")

    # 6. Evaluation, Model Comparison, and ARI Matrix
    logger.info("\n--- STEP 6: EVALUATION & MODEL COMPARISON ---")
    model_comparison_df = pd.DataFrame([km_metrics, db_metrics, agg_metrics])
    model_comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    logger.info(f"\nModel Comparison:\n{model_comparison_df.to_string(index=False)}")

    # ARI Matrix between models
    ari_df = compute_ari_matrix({
        "K-Means": km_labels,
        "DBSCAN": db_labels,
        "Agglomerative": agg_labels
    })
    logger.info(f"\nAdjusted Rand Index Matrix:\n{ari_df.to_string()}")

    # DBSCAN Noise Point Profile (VIP / Wholesale identification)
    noise_eval = profile_dbscan_noise(features_df, db_labels)
    logger.info(f"DBSCAN Noise Profile:\n{noise_eval['summary'].to_string()}")

    # 2D PCA Projection
    X_pca = compute_pca_projection(X_scaled, n_components=2)
    features_df["PCA1"] = X_pca[:, 0]
    features_df["PCA2"] = X_pca[:, 1]
    features_df["KMeans_Cluster"] = km_labels
    features_df["DBSCAN_Cluster"] = db_labels
    features_df["Agglomerative_Cluster"] = agg_labels

    # 7. Name K-Means Segments from Centroid Values
    logger.info("\n--- STEP 7: SEGMENT NAMING & PROFILING ---")
    df_segmented = assign_kmeans_segment_names(features_df, km_labels)
    segment_profiles = compute_segment_profiles(df_segmented, cluster_col="Segment")
    segment_profiles.to_csv(SEGMENT_PROFILES_CSV, index=False)
    logger.info(f"\nSegment Profiles:\n{segment_profiles.to_string(index=False)}")

    # 8. Next-Level Extras
    logger.info("\n--- STEP 8: ADVANCED BUSINESS INTELLIGENCE EXTRAS ---")
    # 8A. Segment Product Affinities & Pareto Share
    seg_insights = compute_segment_product_insights(df_cleaned, df_segmented)
    logger.info(f"Top Product per Segment:\n{seg_insights['top_product_per_segment'].to_string(index=False)}")
    logger.info(f"Pareto Revenue Share per Segment:\n{seg_insights['pareto_summary'].to_string(index=False)}")

    # 8B. Product Catalog Clustering
    product_clusters_df = cluster_products(df_cleaned, n_clusters=4, save_csv=True)
    logger.info(f"Product Tiers distribution:\n{product_clusters_df['ProductTier'].value_counts().to_string()}")

    # 8C. Market Basket Analysis (Association Rules)
    mba_rules, mba_lookup = run_market_basket_analysis(df_cleaned, country="United Kingdom", save_csv=True)
    logger.info(f"Generated {len(mba_rules)} association rules for UK retail transactions.")

    # 9. Save Evaluation Plots
    logger.info("\n--- STEP 9: GENERATE PLOT ARTIFACTS ---")
    save_evaluation_plots(km_search, k_distances, best_eps, X_scaled, figures_dir=FIGURES_DIR)

    # 10. Save Segmented Customers Output
    df_segmented.to_csv(CUSTOMERS_SEGMENTED_CSV, index=False)
    logger.info(f"Saved segmented customers dataset to {CUSTOMERS_SEGMENTED_CSV}")

    # 11. Export Lightweight Deployment Assets
    logger.info("\n--- STEP 11: EXPORT DEPLOYMENT ASSETS ---")
    export_deploy_assets()

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
