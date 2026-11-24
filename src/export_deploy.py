"""
Export utility to generate lightweight, production-ready assets for cloud deployment.
Ensures the dashboard can run completely independently of raw data and large intermediate files.
"""

import logging
import shutil
from pathlib import Path
import pandas as pd
from src.config import (
    CLEANED_DATA_FILE,
    CLEANED_DATA_CSV,
    CUSTOMERS_SEGMENTED_CSV,
    MODEL_COMPARISON_CSV,
    TOP_PRODUCTS_CSV,
    PRODUCT_CLUSTERS_CSV,
    MARKET_BASKET_RULES_CSV,
    SEGMENT_PROFILES_CSV,
    DEPLOY_DATA_DIR,
    DEPLOY_TRANSACTIONS_FILE,
    DEPLOY_CUSTOMERS_FILE,
    DEPLOY_MODEL_COMPARISON_FILE,
    DEPLOY_TOP_PRODUCTS_FILE,
    DEPLOY_PRODUCT_CLUSTERS_FILE,
    DEPLOY_MARKET_BASKET_RULES_FILE,
    DEPLOY_SEGMENT_PROFILES_FILE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_deploy_assets(deploy_dir: Path = DEPLOY_DATA_DIR) -> None:
    """
    Creates optimized, memory-efficient deployment dataset in data/deploy/.
    - Downcasts dtypes on transactions for fast parsing and sub-5MB Parquet footprint.
    - Copies all precomputed analytical tables (segments, rules, clusters, benchmarks).
    """
    deploy_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting deployment assets to {deploy_dir}...")

    # 1. Optimize transactions dataframe
    if CLEANED_DATA_FILE.exists():
        df = pd.read_parquet(CLEANED_DATA_FILE)
    else:
        df = pd.read_csv(CLEANED_DATA_CSV)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Keep essential columns and downcast types
    keep_cols = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country", "Revenue"
    ]
    df_compact = df[[c for c in keep_cols if c in df.columns]].copy()
    
    # Efficient dtypes
    df_compact["Quantity"] = df_compact["Quantity"].astype("int32")
    df_compact["UnitPrice"] = df_compact["UnitPrice"].astype("float32")
    df_compact["Revenue"] = df_compact["Revenue"].astype("float32")
    df_compact["Country"] = df_compact["Country"].astype("category")
    if "CustomerID" in df_compact.columns:
        df_compact["CustomerID"] = df_compact["CustomerID"].astype(str)

    # Save compact parquet
    df_compact.to_parquet(DEPLOY_TRANSACTIONS_FILE, index=False, compression="snappy")
    parquet_size_mb = DEPLOY_TRANSACTIONS_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved compact transactions parquet ({parquet_size_mb:.2f} MB) -> {DEPLOY_TRANSACTIONS_FILE}")

    # 2. Copy companion CSV tables
    companions = [
        (CUSTOMERS_SEGMENTED_CSV, DEPLOY_CUSTOMERS_FILE),
        (MODEL_COMPARISON_CSV, DEPLOY_MODEL_COMPARISON_FILE),
        (TOP_PRODUCTS_CSV, DEPLOY_TOP_PRODUCTS_FILE),
        (PRODUCT_CLUSTERS_CSV, DEPLOY_PRODUCT_CLUSTERS_FILE),
        (MARKET_BASKET_RULES_CSV, DEPLOY_MARKET_BASKET_RULES_FILE),
        (SEGMENT_PROFILES_CSV, DEPLOY_SEGMENT_PROFILES_FILE)
    ]

    total_bytes = DEPLOY_TRANSACTIONS_FILE.stat().st_size
    for src_path, dst_path in companions:
        if src_path.exists():
            shutil.copyfile(src_path, dst_path)
            size_kb = dst_path.stat().st_size / 1024
            total_bytes += dst_path.stat().st_size
            logger.info(f"Copied {src_path.name} ({size_kb:.1f} KB) -> {dst_path.name}")
        else:
            logger.warning(f"Companion file {src_path} not found.")

    total_mb = total_bytes / (1024 * 1024)
    logger.info(f"Deployment assets export complete! Total deploy folder size: {total_mb:.2f} MB")


if __name__ == "__main__":
    export_deploy_assets()
