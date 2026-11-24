"""
Global configuration and paths for the Retail Customer Segmentation project.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEPLOY_DATA_DIR = DATA_DIR / "deploy"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# File paths
RAW_DATA_FILE = RAW_DATA_DIR / "data.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_retail_data.parquet"
CLEANED_DATA_CSV = PROCESSED_DATA_DIR / "cleaned_retail_data.csv"
CUSTOMER_FEATURES_FILE = PROCESSED_DATA_DIR / "customer_features.parquet"
CUSTOMERS_SEGMENTED_CSV = PROCESSED_DATA_DIR / "customers_with_segments.csv"
MODEL_COMPARISON_CSV = PROCESSED_DATA_DIR / "model_comparison.csv"
TOP_PRODUCTS_CSV = PROCESSED_DATA_DIR / "top_products.csv"
MARKET_BASKET_RULES_CSV = PROCESSED_DATA_DIR / "market_basket_rules.csv"
PRODUCT_CLUSTERS_CSV = PROCESSED_DATA_DIR / "product_clusters.csv"
SEGMENT_PROFILES_CSV = PROCESSED_DATA_DIR / "segment_profiles.csv"

# Deploy file paths
DEPLOY_TRANSACTIONS_FILE = DEPLOY_DATA_DIR / "cleaned_transactions.parquet"
DEPLOY_CUSTOMERS_FILE = DEPLOY_DATA_DIR / "customers_with_segments.csv"
DEPLOY_MODEL_COMPARISON_FILE = DEPLOY_DATA_DIR / "model_comparison.csv"
DEPLOY_TOP_PRODUCTS_FILE = DEPLOY_DATA_DIR / "top_products.csv"
DEPLOY_PRODUCT_CLUSTERS_FILE = DEPLOY_DATA_DIR / "product_clusters.csv"
DEPLOY_MARKET_BASKET_RULES_FILE = DEPLOY_DATA_DIR / "market_basket_rules.csv"
DEPLOY_SEGMENT_PROFILES_FILE = DEPLOY_DATA_DIR / "segment_profiles.csv"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DEPLOY_DATA_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for reproducibility
RANDOM_STATE = 42

# Known non-product stock codes to filter out
NON_PRODUCT_STOCK_CODES = {
    "POST", "D", "M", "DOT", "BANK CHARGES", "CRUK", "AMAZONFEE",
    "PADS", "S", "B", "gift_0001_10", "gift_0001_20", "gift_0001_30",
    "gift_0001_40", "gift_0001_50", "DCGS0003", "DCGS0004", "DCGS0069",
    "DCGS0070", "DCGS0076", "DCGSSBOY", "DCGSSGIRL"
}

# RFM & Feature Settings
FEATURE_COLUMNS = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue",
    "UniqueProducts", "AvgBasketSize"
]
LOG_TRANSFORM_COLUMNS = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue",
    "UniqueProducts", "AvgBasketSize"
]

# Clustering Settings
KMEANS_K_RANGE = range(2, 11)
DEFAULT_KMEANS_K = 4
DBSCAN_DEFAULT_EPS = 0.8
DBSCAN_DEFAULT_MIN_SAMPLES = 10
AGGLOMERATIVE_DEFAULT_CLUSTERS = 4
AGGLOMERATIVE_LINKAGES = ["ward", "complete", "average"]

# Segment Name Heuristics for K-Means (based on RFM centroids)
SEGMENT_LABELS = {
    "Champions": "High Monetary, High Frequency, Very Low Recency",
    "Loyal": "Moderate-High Monetary & Frequency, Low Recency",
    "Potential Loyalists": "Recent buyers with average spend",
    "At-Risk": "Previously frequent/high spenders who haven't purchased recently",
    "Lost / Hibernating": "Low Monetary, Low Frequency, High Recency"
}
