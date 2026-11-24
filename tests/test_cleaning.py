"""
Unit tests for data cleaning logic, business rules, and cancellation netting.
"""

import pandas as pd
import pytest
from src.cleaning import clean_retail_data, net_out_cancellations, get_customer_clustering_data


@pytest.fixture
def sample_raw_data():
    """Returns a synthetic raw dataframe with cancellation scenarios and edge cases."""
    data = {
        "InvoiceNo": [
            "536365",       # Normal order A: qty 6
            "536365",       # Exact duplicate row of normal order A
            "536366",       # Normal order B: qty 10 (will be partially cancelled)
            "536367",       # Normal order C: qty 5 (will be fully cancelled)
            "C536368",      # Cancellation for B: qty -3 (partial)
            "C536369",      # Cancellation for C: qty -5 (full)
            "536370",       # Invalid negative quantity without order
            "536371",       # Zero unit price
            "536372",       # Non-product code POST
            "536373",       # Valid product with missing customer ID
            "536374"        # Valid product with customer ID
        ],
        "StockCode": [
            "85123A",
            "85123A",
            "22423",
            "22197",
            "22423",
            "22197",
            "84077",
            "22423",
            "POST",
            "22423",
            "22423"
        ],
        "Description": [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "REGENCY CAKESTAND 3 TIER",
            "POPCORN HOLDER",
            "REGENCY CAKESTAND 3 TIER",
            "POPCORN HOLDER",
            "WORLD WAR 2 GLIDERS",
            "REGENCY CAKESTAND 3 TIER",
            "POSTAGE",
            "REGENCY CAKESTAND 3 TIER",
            "REGENCY CAKESTAND 3 TIER"
        ],
        "Quantity": [6, 6, 10, 5, -3, -5, -2, 2, 1, 4, 8],
        "InvoiceDate": [
            "2010-12-01 08:26:00",
            "2010-12-01 08:26:00",
            "2010-12-01 08:30:00",
            "2010-12-01 08:32:00",
            "2010-12-01 08:35:00",
            "2010-12-01 08:40:00",
            "2010-12-01 08:42:00",
            "2010-12-01 08:45:00",
            "2010-12-01 08:50:00",
            "2010-12-01 08:55:00",
            "2010-12-01 09:00:00"
        ],
        "UnitPrice": [2.55, 2.55, 12.75, 0.85, 12.75, 0.85, 0.29, 0.0, 18.0, 12.75, 12.75],
        "CustomerID": [17850.0, 17850.0, 13047.0, 12583.0, 13047.0, 12583.0, 17850.0, 13047.0, 12583.0, None, 12583.0],
        "Country": ["United Kingdom"] * 11
    }
    return pd.DataFrame(data)


def test_clean_retail_data(sample_raw_data, tmp_path):
    out_parquet = tmp_path / "clean.parquet"
    out_csv = tmp_path / "clean.csv"

    cleaned_df, counts_log = clean_retail_data(
        sample_raw_data,
        save_processed=True,
        output_parquet=out_parquet,
        output_csv=out_csv
    )

    # 1. Check duplicate dropped: only 1 instance of 536365 / 85123A
    assert len(cleaned_df[cleaned_df["InvoiceNo"] == "536365"]) == 1

    # 2. Check full cancellation: order 536367 (POPCORN HOLDER, qty 5) was fully netted out by C536369 (qty -5)
    assert not (cleaned_df["InvoiceNo"] == "536367").any()

    # 3. Check partial cancellation: order 536366 (REGENCY CAKESTAND, initial qty 10) reduced by 3 to qty 7
    part_order = cleaned_df[cleaned_df["InvoiceNo"] == "536366"]
    assert len(part_order) == 1
    assert part_order.iloc[0]["Quantity"] == 7
    assert part_order.iloc[0]["Revenue"] == pytest.approx(7 * 12.75)

    # 4. Check no cancelled invoices remain in cleaned output
    assert not cleaned_df["InvoiceNo"].str.startswith("C").any()

    # 5. Check invalid Quantity and UnitPrice removed
    assert (cleaned_df["Quantity"] > 0).all()
    assert (cleaned_df["UnitPrice"] > 0).all()

    # 6. Check non-product code POST removed
    assert not cleaned_df["StockCode"].isin(["POST"]).any()

    # 7. Check Revenue calculation
    assert "Revenue" in cleaned_df.columns
    assert (cleaned_df["Revenue"] == cleaned_df["Quantity"] * cleaned_df["UnitPrice"]).all()

    # 8. Check missing CustomerID row preserved (536373)
    assert cleaned_df["CustomerID"].isna().any()

    # 9. Verify netting metrics logged
    assert counts_log["4_netting_original_rows_removed"] >= 1
    assert counts_log["4_netting_original_rows_reduced"] >= 1


def test_get_customer_clustering_data(sample_raw_data, tmp_path):
    cleaned_df, _ = clean_retail_data(sample_raw_data, save_processed=False)
    cust_df = get_customer_clustering_data(cleaned_df)

    # Missing CustomerID rows must be dropped for customer clustering
    assert not cust_df["CustomerID"].isna().any()
