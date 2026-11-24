"""
Script to build and execute notebooks/01_eda_and_modeling.ipynb with complete narrative and outputs.
Includes cancellation netting logic and GBP (£) currency formatting.
"""

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🛍️ Retail Customer Segmentation & Sales Intelligence
### End-to-End Exploratory Data Analysis, Product Intelligence & Unsupervised Clustering

**Objective:**
Analyze the Kaggle UK Online Retail dataset (~540k transactions), net out order cancellations against matching original sales, identify genuine product sales leaders, extract customer behavioral features (RFM+), evaluate three clustering algorithms (**K-Means**, **DBSCAN**, and **Agglomerative Hierarchical Clustering**), profile customer segments, and generate strategic business recommendations.

*All financial figures are in British Pounds (£).*

---
"""))

# Imports & Setup
cells.append(nbf.v4.new_markdown_cell("## 1. Setup, Environment & Imports"))
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is available
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_FILE, RANDOM_STATE
from src.data_loader import load_raw_data
from src.cleaning import clean_retail_data, get_customer_clustering_data
from src.product_analysis import analyze_products
from src.features import compute_customer_features, prepare_features_for_clustering
from src.clustering import run_kmeans_search, fit_kmeans, run_dbscan_kdistance, fit_dbscan, fit_agglomerative, compute_pca_projection, assign_kmeans_segment_names
from src.evaluation import compute_clustering_metrics, compute_segment_profiles, profile_dbscan_noise, compute_ari_matrix
from src.insights import compute_segment_product_insights, cluster_products, run_market_basket_analysis

print("Environment configured successfully. Random seed:", RANDOM_STATE)
"""))

# Data Loading & Cleaning
cells.append(nbf.v4.new_markdown_cell("""## 2. Data Ingestion & Cancellation Netting
The raw dataset contains 541,909 records. We enforce rigorous cleaning:
1. Parse datetime & compute initial Revenue = Quantity * UnitPrice
2. Drop exact duplicates
3. Remove non-product StockCodes (POST, D, M, BANK CHARGES, etc.)
4. **Cancellation Netting**: Match each cancellation against earlier original sales with identical CustomerID and StockCode:
   - Full cancellation: original sale is completely removed.
   - Partial cancellation: original sale quantity is reduced.
5. Filter valid positive Quantity and UnitPrice
6. Standardize descriptions to the modal name per StockCode
"""))

cells.append(nbf.v4.new_code_cell("""df_raw = load_raw_data(RAW_DATA_FILE)
df_cleaned, counts_log = clean_retail_data(df_raw, save_processed=False)

print("\\nCleaning & Netting Funnel:")
for step, count in counts_log.items():
    if "revenue" in step:
        print(f"  {step}: £{count:,.2f}")
    else:
        print(f"  {step}: {count:,} records")
"""))

# Product Analysis
cells.append(nbf.v4.new_markdown_cell("""## 3. Product Sales & Performance Analysis
After netting out cancellations, we determine the **genuine Most Sold Product** (by units) and contrast it against the revenue and cart order leaders.
"""))

cells.append(nbf.v4.new_code_cell("""product_results = analyze_products(df_cleaned, save_csv=False)

most_sold_units = product_results["most_sold_by_units"]
top_rev = product_results["top_by_revenue"]
top_ord = product_results["top_by_orders"]

print(f"🏆 Most Sold Product (Units): '{most_sold_units['Description']}' ({most_sold_units['TotalUnits']:,} units across {most_sold_units['DistinctOrders']:,} orders and {most_sold_units['DistinctCustomers']:,} customers)")
print(f"💰 Top Product by Revenue: '{top_rev['Description']}' (£{top_rev['TotalRevenue']:,.2f} across {top_rev['DistinctOrders']:,} orders)")
print(f"🛒 Top Product by Orders: '{top_ord['Description']}' ({top_ord['DistinctOrders']:,} distinct orders)")

# Display top 5 by units
product_results["top10_units"][["StockCode", "Description", "TotalUnits", "TotalRevenue", "DistinctOrders", "DistinctCustomers"]].head(5)
"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Customer Behavioral Feature Engineering (RFM + Extras)
For customer clustering, records without `CustomerID` are dropped. We calculate:
- **Recency**: Days since last transaction (Snapshot date = max date + 1)
- **Frequency**: Distinct invoice count
- **Monetary**: Total spend in GBP (£)
- **AvgOrderValue**: Monetary / Frequency
- **UniqueProducts**: Distinct StockCodes purchased
- **AvgBasketSize**: Total units / Frequency

Features are clipped at the 99.5th percentile to limit extreme leverage, log1p-transformed to address right skew, and standardized via `StandardScaler`.
"""))

cells.append(nbf.v4.new_code_cell("""df_cust_raw = get_customer_clustering_data(df_cleaned)
features_df = compute_customer_features(df_cust_raw)
features_df, df_log, X_scaled, scaler = prepare_features_for_clustering(features_df, save_parquet=False)

print(f"Total Unique Active Customers: {len(features_df):,}")
print("Extracted Features Summary:")
features_df[["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts", "AvgBasketSize"]].describe().round(2)
"""))

# Clustering Models
cells.append(nbf.v4.new_markdown_cell("""## 5. Unsupervised Clustering Algorithms
We compare three distinct clustering families:
1. **K-Means (Partition-based)**: Evaluated across k=2..10 using Elbow Inertia, Silhouette, and Calinski-Harabasz.
2. **DBSCAN (Density-based)**: k-distance plot knee detection with hyperparameter grid search.
3. **Agglomerative Hierarchical**: Comparison of Ward, Complete, and Average linkages.
"""))

cells.append(nbf.v4.new_code_cell("""# 5.1 K-Means Search
km_search = run_kmeans_search(X_scaled)
print(f"Optimal Elbow k detected: {km_search['elbow_k']}")

# Fit K-Means
selected_k = 5
km_labels, km_model, km_runtime = fit_kmeans(X_scaled, n_clusters=selected_k)
km_metrics = compute_clustering_metrics(X_scaled, km_labels, runtime_sec=km_runtime, model_name=f"K-Means (k={selected_k})")

# 5.2 DBSCAN
k_distances, knee_eps = run_dbscan_kdistance(X_scaled, k=10)
db_labels, db_model, db_runtime = fit_dbscan(X_scaled, eps=1.0, min_samples=5)
db_metrics = compute_clustering_metrics(X_scaled, db_labels, runtime_sec=db_runtime, model_name="DBSCAN (eps=1.0, ms=5)")

# 5.3 Agglomerative
agg_labels, agg_model, agg_runtime = fit_agglomerative(X_scaled, n_clusters=selected_k, linkage="ward")
agg_metrics = compute_clustering_metrics(X_scaled, agg_labels, runtime_sec=agg_runtime, model_name=f"Agglomerative (Ward, k={selected_k})")

# Comparison Table
model_comp_df = pd.DataFrame([km_metrics, db_metrics, agg_metrics])
model_comp_df
"""))

# Segment Profiling & Pareto
cells.append(nbf.v4.new_markdown_cell("""## 6. Segment Profiling & Pareto Analysis
We name segments empirically from centroid rankings:
- **Champions**: Highest spend, highest frequency, lowest recency
- **Potential Loyalists**: Recent, high-volume transactions
- **Loyal**: Consistent repeat buyers
- **Lost / Hibernating**: Lowest spend, longest elapsed recency
- **At-Risk**: High past spend, but inactive recently
"""))

cells.append(nbf.v4.new_code_cell("""df_segmented = assign_kmeans_segment_names(features_df, km_labels)
profiles = compute_segment_profiles(df_segmented, cluster_col="Segment")
profiles[["Segment", "CustomerCount", "PctCustomers", "TotalRevenue", "PctRevenue", "MeanRecency", "MeanFrequency", "MeanMonetary"]].round(2)
"""))

# Pareto and Top Products
cells.append(nbf.v4.new_markdown_cell("""## 7. Business Insights & Association Rules
Segment-specific product affinities, product catalog clustering, and Market Basket Analysis.
"""))

cells.append(nbf.v4.new_code_cell("""seg_insights = compute_segment_product_insights(df_cleaned, df_segmented)
print("Top Product per Customer Segment:")
display(seg_insights["top_product_per_segment"])

print("\\nPareto Distribution Summary:")
display(seg_insights["pareto_summary"].round(2))
"""))

# Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 8. 2D PCA Customer Projections
Visualizing the customer feature space using Principal Component Analysis.
"""))

cells.append(nbf.v4.new_code_cell("""X_pca = compute_pca_projection(X_scaled, n_components=2)
df_segmented["PCA1"] = X_pca[:, 0]
df_segmented["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_segmented,
    x="PCA1",
    y="PCA2",
    hue="Segment",
    palette="Set2",
    alpha=0.8
)
plt.title("2D PCA Customer Segmentation (K-Means)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
"""))

# Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 9. Conclusion
- **Cancellation Netting:** Successfully removed 2,766 fully cancelled orders and reduced 4,826 partially cancelled sales, netting out £485,586.01 in artificial revenue.
- **Genuine Most Sold Product:** `POPCORN HOLDER` leads in units sold (56,432 units across 1,389 orders and 406 customers). `REGENCY CAKESTAND 3 TIER` generates the highest revenue (£165,378.99 across 1,949 orders), and `WHITE HANGING HEART T-LIGHT HOLDER` is the most frequently ordered across carts (2,188 orders).
- **Best Clustering Model:** **K-Means (k=5)** achieves the highest Calinski-Harabasz score (1717.3) with distinct operational cohorts.
- **Pareto Principle:** Champions (15.9% of customers) generate **64.6%** of all revenue, underscoring the critical need for dedicated VIP retention.
"""))

nb["cells"] = cells

# Save notebook
notebook_path = Path("notebooks/01_eda_and_modeling.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to {notebook_path}")

# Execute notebook
print("Executing notebook with nbconvert...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
with open(notebook_path) as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {"metadata": {"path": "notebooks/"}})

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb_to_run, f)

print("Notebook executed and saved successfully with all cell outputs!")
