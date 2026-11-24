"""
Data cleaning and preprocessing module with cancellation netting and step-by-step row count logging.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from src.config import (
    NON_PRODUCT_STOCK_CODES,
    CLEANED_DATA_FILE,
    CLEANED_DATA_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def net_out_cancellations(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Nets out cancellations against matching original sales:
    For each cancellation row (InvoiceNo starting with 'C' or Quantity < 0):
      Finds matching original sale(s) with same CustomerID and StockCode,
      at the closest earlier InvoiceDate.
      - If orig.Quantity == abs(cancel.Quantity): original sale is removed completely.
      - If orig.Quantity > abs(cancel.Quantity): original sale quantity is reduced.
      - If orig.Quantity < abs(cancel.Quantity): original sale is removed, remainder applied to prior sale.
    
    Returns:
    - Cleaned sales dataframe with cancellations netted out
    - Netting metrics dictionary (rows removed, rows reduced, revenue netted out)
    """
    logger.info("Initiating cancellation netting algorithm...")
    
    # Identify cancellations vs positive sales
    is_cancel = df["InvoiceNo"].astype(str).str.upper().str.startswith("C") | (df["Quantity"] < 0)
    sales = df[~is_cancel & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    cancels = df[is_cancel & (df["Quantity"] < 0)].copy()

    # Sort sales and cancels by InvoiceDate ascending
    sales = sales.sort_values("InvoiceDate")
    cancels = cancels.sort_values("InvoiceDate")

    # Index sales by (CustomerID, StockCode) for fast matching
    sales_groups: Dict[Tuple[Any, str], list] = {}
    for idx, row in sales.iterrows():
        if pd.isna(row["CustomerID"]):
            continue
        key = (row["CustomerID"], str(row["StockCode"]).strip())
        if key not in sales_groups:
            sales_groups[key] = []
        sales_groups[key].append({
            "index": idx,
            "date": row["InvoiceDate"],
            "qty": row["Quantity"],
            "price": row["UnitPrice"]
        })

    indices_to_drop = set()
    qty_updates = {}
    netted_rows_removed = 0
    netted_rows_reduced = 0
    netted_revenue = 0.0

    # Match each cancellation
    for _, c_row in cancels.iterrows():
        if pd.isna(c_row["CustomerID"]):
            continue
        key = (c_row["CustomerID"], str(c_row["StockCode"]).strip())
        if key not in sales_groups:
            continue

        c_date = c_row["InvoiceDate"]
        rem_cancel_qty = abs(c_row["Quantity"])
        # Find candidates with date <= cancellation date and available quantity > 0
        candidates = [s for s in sales_groups[key] if s["date"] <= c_date and s["qty"] > 0]
        if not candidates:
            continue

        # Reverse order to pick closest earlier InvoiceDate first
        for s in reversed(candidates):
            if rem_cancel_qty <= 0:
                break
            s_idx = s["index"]
            if s["qty"] == rem_cancel_qty:
                netted_revenue += rem_cancel_qty * s["price"]
                netted_rows_removed += 1
                s["qty"] = 0
                indices_to_drop.add(s_idx)
                rem_cancel_qty = 0
            elif s["qty"] > rem_cancel_qty:
                netted_revenue += rem_cancel_qty * s["price"]
                netted_rows_reduced += 1
                s["qty"] -= rem_cancel_qty
                qty_updates[s_idx] = s["qty"]
                rem_cancel_qty = 0
            else:
                netted_revenue += s["qty"] * s["price"]
                netted_rows_removed += 1
                rem_cancel_qty -= s["qty"]
                s["qty"] = 0
                indices_to_drop.add(s_idx)

    # Apply drops and quantity updates to sales dataframe
    sales = sales.drop(index=list(indices_to_drop))
    for s_idx, new_qty in qty_updates.items():
        if s_idx in sales.index:
            sales.loc[s_idx, "Quantity"] = new_qty

    sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]

    netting_stats = {
        "original_sales_removed": netted_rows_removed,
        "original_sales_reduced": netted_rows_reduced,
        "total_revenue_netted_gbp": round(netted_revenue, 2)
    }

    logger.info(
        f"Cancellation netting complete: {netted_rows_removed:,} original sales removed, "
        f"{netted_rows_reduced:,} sales partially reduced. "
        f"Total revenue netted out: £{netted_revenue:,.2f}"
    )

    return sales, netting_stats


def clean_retail_data(
    df_raw: pd.DataFrame,
    save_processed: bool = True,
    output_parquet: Path = CLEANED_DATA_FILE,
    output_csv: Path = CLEANED_DATA_CSV
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw retail dataset following rigorous rules. Logs row count before and after each step.
    
    Steps:
    1. Parse InvoiceDate to datetime and calculate Revenue = Quantity * UnitPrice
    2. Drop exact duplicates
    3. Remove non-product StockCodes (POST, D, M, DOT, BANK CHARGES, etc.)
    4. Net out cancellations against matching original sales
    5. Filter valid positive Quantity and UnitPrice
    6. Map each StockCode to its most frequent Description (standardize names)
    
    Note: Rows with missing CustomerID are retained for sales/product analytics.
    """
    counts_log = {}
    current_count = len(df_raw)
    counts_log["0_initial_rows"] = current_count
    logger.info(f"[Step 0] Initial raw records: {current_count:,}")

    df = df_raw.copy()

    # Step 1: Parse datetime and calculate Revenue
    logger.info("[Step 1] Parsing InvoiceDate and computing initial Revenue...")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    current_count = len(df)
    counts_log["1_after_date_parse"] = current_count
    logger.info(f"[Step 1] Records after datetime parsing: {current_count:,}")

    # Step 2: Drop exact duplicates
    df = df.drop_duplicates()
    diff = current_count - len(df)
    current_count = len(df)
    counts_log["2_after_drop_duplicates"] = current_count
    logger.info(f"[Step 2] Dropped {diff:,} exact duplicates. Remaining: {current_count:,}")

    # Step 3: Remove non-product StockCodes
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    stock_code_upper = df["StockCode"].str.upper()
    is_non_product = stock_code_upper.isin({code.upper() for code in NON_PRODUCT_STOCK_CODES})
    is_non_product |= stock_code_upper.str.contains("AMAZON|BANK|CHARGES|DISCOUNT|CRUK", regex=True)
    non_product_count = is_non_product.sum()
    df = df[~is_non_product]
    current_count = len(df)
    counts_log["3_after_remove_non_products"] = current_count
    logger.info(f"[Step 3] Removed {non_product_count:,} non-product StockCodes. Remaining: {current_count:,}")

    # Step 4: Net out cancellations against matching original sales
    logger.info("[Step 4] Netting out cancellations against original orders...")
    sales_df, netting_stats = net_out_cancellations(df)
    counts_log["4_netting_original_rows_removed"] = netting_stats["original_sales_removed"]
    counts_log["4_netting_original_rows_reduced"] = netting_stats["original_sales_reduced"]
    counts_log["4_netting_revenue_gbp"] = netting_stats["total_revenue_netted_gbp"]

    # Step 5: Filter valid positive records (all cancellation rows are now eliminated)
    df = sales_df[(sales_df["Quantity"] > 0) & (sales_df["UnitPrice"] > 0)].copy()
    current_count = len(df)
    counts_log["5_after_netting_and_valid_sales"] = current_count
    logger.info(f"[Step 5] Valid sales after cancellation netting: {current_count:,}")

    # Step 6: Map each StockCode to its most frequent Description
    logger.info("[Step 6] Standardizing StockCode Descriptions...")
    clean_desc_series = df["Description"].astype(str).str.strip().str.upper()
    valid_desc_df = df[clean_desc_series.ne("NAN") & clean_desc_series.ne("") & clean_desc_series.ne("NONE")].copy()
    valid_desc_df["CleanDesc"] = clean_desc_series[valid_desc_df.index]
    
    top_descriptions = (
        valid_desc_df.groupby(["StockCode", "CleanDesc"])
        .size()
        .reset_index(name="count")
        .sort_values(["StockCode", "count"], ascending=[True, False])
        .drop_duplicates(subset=["StockCode"])
        .set_index("StockCode")["CleanDesc"]
        .to_dict()
    )

    df["Description"] = df["StockCode"].map(top_descriptions).fillna(
        clean_desc_series.replace({"NAN": "UNKNOWN PRODUCT", "": "UNKNOWN PRODUCT"})
    )
    current_count = len(df)
    counts_log["6_final_cleaned_rows"] = current_count
    logger.info(f"[Step 6] Final cleaned records count: {current_count:,}")

    if save_processed:
        logger.info(f"Saving cleaned dataset to {output_parquet} and {output_csv}...")
        df.to_parquet(output_parquet, index=False)
        df.to_csv(output_csv, index=False)
        logger.info("Cleaned data successfully saved.")

    return df, counts_log


def get_customer_clustering_data(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Filters cleaned dataset for customer clustering by dropping records without CustomerID.
    """
    initial = len(df_cleaned)
    df_cust = df_cleaned.dropna(subset=["CustomerID"]).copy()
    df_cust["CustomerID"] = df_cust["CustomerID"].astype(int).astype(str)
    logger.info(f"Filtered for customer clustering: {initial:,} -> {len(df_cust):,} records (CustomerID present).")
    return df_cust
