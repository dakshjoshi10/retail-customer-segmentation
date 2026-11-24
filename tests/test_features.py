"""
Unit tests for customer feature engineering and RFM calculations.
"""

import numpy as np
import pandas as pd
import pytest
from src.features import compute_customer_features, prepare_features_for_clustering


@pytest.fixture
def sample_cleaned_data():
    """Provides cleaned transaction data for two customers."""
    data = {
        "InvoiceNo": ["1001", "1002", "1003", "1004"],
        "StockCode": ["A1", "A2", "A1", "B1"],
        "Description": ["PROD A", "PROD B", "PROD A", "PROD C"],
        "Quantity": [2, 3, 5, 10],
        "InvoiceDate": [
            pd.Timestamp("2021-01-01 10:00:00"),
            pd.Timestamp("2021-01-05 10:00:00"),
            pd.Timestamp("2021-01-10 10:00:00"),
            pd.Timestamp("2021-01-12 10:00:00")
        ],
        "UnitPrice": [10.0, 20.0, 10.0, 5.0],
        "CustomerID": [101.0, 101.0, 101.0, 102.0],
        "Country": ["United Kingdom"] * 4
    }
    df = pd.DataFrame(data)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def test_compute_customer_features(sample_cleaned_data):
    snapshot_date = pd.Timestamp("2021-01-13 10:00:00")
    features = compute_customer_features(sample_cleaned_data, snapshot_date=snapshot_date)

    assert len(features) == 2

    # Check Customer 101:
    # Invoices: 1001, 1002, 1003 -> Frequency = 3
    # Quantities: 2, 3, 5 -> TotalUnits = 10
    # Revenues: 20, 60, 50 -> Monetary = 130
    # Last purchase: 2021-01-10 -> Recency = (2021-01-13 - 2021-01-10) = 3 days
    # AvgOrderValue: 130 / 3 = 43.333...
    # UniqueProducts: 'A1', 'A2' -> 2
    # AvgBasketSize: 10 / 3 = 3.333...
    c101 = features[features["CustomerID"] == "101"].iloc[0]
    assert c101["Frequency"] == 3
    assert c101["Monetary"] == 130.0
    assert c101["Recency"] == 3
    assert pytest.approx(c101["AvgOrderValue"], 0.01) == 43.333
    assert c101["UniqueProducts"] == 2
    assert pytest.approx(c101["AvgBasketSize"], 0.01) == 3.333

    # Check Customer 102:
    # Frequency = 1, Monetary = 50, Recency = 1 day, UniqueProducts = 1
    c102 = features[features["CustomerID"] == "102"].iloc[0]
    assert c102["Frequency"] == 1
    assert c102["Monetary"] == 50.0
    assert c102["Recency"] == 1
    assert c102["AvgOrderValue"] == 50.0
    assert c102["UniqueProducts"] == 1


def test_prepare_features_for_clustering(sample_cleaned_data):
    features = compute_customer_features(sample_cleaned_data)
    features_df, df_log, X_scaled, scaler = prepare_features_for_clustering(
        features,
        save_parquet=False
    )

    # Check output shapes
    assert X_scaled.shape == (2, 6)
    assert not np.isnan(X_scaled).any()

    # Standard scaling check: mean approximately 0 and unit variance across features
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-7)
