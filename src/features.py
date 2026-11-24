"""
Customer RFM and behavioral feature engineering module.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import (
    FEATURE_COLUMNS,
    CUSTOMER_FEATURES_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_customer_features(df_cleaned: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Computes RFM and extended customer-level behavioral metrics:
    - Recency: Days between customer's last purchase and snapshot_date (max date + 1 day)
    - Frequency: Count of distinct invoices (transactions)
    - Monetary: Total monetary spend (sum of Revenue)
    - AvgOrderValue: Monetary / Frequency
    - UniqueProducts: Number of distinct StockCodes purchased
    - AvgBasketSize: Total units purchased / Frequency
    """
    # Filter only rows with valid CustomerID
    df_valid = df_cleaned.dropna(subset=["CustomerID"]).copy()
    df_valid["CustomerID"] = df_valid["CustomerID"].astype(int).astype(str)

    if snapshot_date is None:
        snapshot_date = df_valid["InvoiceDate"].max() + pd.Timedelta(days=1)
    
    logger.info(f"Computing customer features with snapshot date: {snapshot_date}")

    # Group by CustomerID
    features = (
        df_valid.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
            TotalUnits=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique")
        )
        .reset_index()
    )

    # Calculate Recency in days
    features["Recency"] = (snapshot_date - features["LastPurchase"]).dt.days

    # Calculate Derived Ratios
    features["AvgOrderValue"] = features["Monetary"] / features["Frequency"]
    features["AvgBasketSize"] = features["TotalUnits"] / features["Frequency"]

    # Filter out customers with non-positive monetary spend if any remain
    features = features[features["Monetary"] > 0].copy()

    # Retain Country (primary country of the customer)
    customer_country = (
        df_valid.groupby("CustomerID")["Country"]
        .agg(lambda x: x.mode().iloc[0] if not x.empty else "Unknown")
        .reset_index()
    )
    features = features.merge(customer_country, on="CustomerID", how="left")

    logger.info(f"Generated features for {len(features):,} unique active customers.")
    return features


def prepare_features_for_clustering(
    features_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    outlier_quantile: float = 0.995,
    save_parquet: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Transforms skewed customer metrics for clustering algorithms:
    1. Winsorization / capping at high quantile (e.g., 99.5%) to limit extreme leverage points
    2. log1p transformation to mitigate heavy right-skew in financial and frequency data
    3. StandardScaler normalization to ensure zero mean and unit variance across dimensions
    
    Returns:
    - features_df: Original unscaled customer features
    - features_transformed: Capped and log1p-transformed features dataframe
    - X_scaled: Scaled numpy matrix ready for distance-based clustering
    - scaler: Fitted StandardScaler instance
    """
    logger.info(f"Preparing features for clustering across columns: {feature_cols}")
    df_trans = features_df[feature_cols].copy()

    # Step 1: Winsorization / Outlier Capping at upper quantile
    for col in feature_cols:
        upper_limit = df_trans[col].quantile(outlier_quantile)
        df_trans[col] = np.clip(df_trans[col], a_min=0, a_max=upper_limit)

    # Step 2: Log1p Transformation
    # Log1p is ideal because all metrics are >= 0 and right-skewed
    df_log = np.log1p(df_trans)

    # Step 3: Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log)

    logger.info(f"Scaled feature matrix shape: {X_scaled.shape}")

    if save_parquet:
        features_df.to_parquet(CUSTOMER_FEATURES_FILE, index=False)
        logger.info(f"Saved customer features to {CUSTOMER_FEATURES_FILE}")

    return features_df, df_log, X_scaled, scaler
