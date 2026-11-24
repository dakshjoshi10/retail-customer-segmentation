"""
Strategic business insights: Segment-product mappings, product clustering,
market basket analysis (association rules), and automated business recommendations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from src.config import (
    RANDOM_STATE,
    MARKET_BASKET_RULES_CSV,
    PRODUCT_CLUSTERS_CSV
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_segment_product_insights(
    df_cleaned: pd.DataFrame,
    df_customers_segmented: pd.DataFrame
) -> Dict[str, Any]:
    """
    Computes:
    1. Most-bought product (by units and revenue) for EACH customer segment.
    2. Revenue share and customer share per segment (Pareto distribution analysis).
    """
    logger.info("Computing segment-specific product affinities and revenue shares...")
    # Filter rows with CustomerID and ensure string type for consistent merging
    df_valid = df_cleaned.dropna(subset=["CustomerID"]).copy()
    df_valid["CustomerID"] = df_valid["CustomerID"].astype(int).astype(str)

    df_customers = df_customers_segmented.copy()
    df_customers["CustomerID"] = df_customers["CustomerID"].astype(str)

    df_merged = df_valid.merge(
        df_customers[["CustomerID", "Segment"]],
        on="CustomerID",
        how="inner"
    )

    # Top product per segment by units
    seg_product_agg = (
        df_merged.groupby(["Segment", "StockCode", "Description"], as_index=False)
        .agg(
            UnitsPurchased=("Quantity", "sum"),
            SegmentRevenue=("Revenue", "sum")
        )
    )

    top_product_by_segment = (
        seg_product_agg.sort_values(["Segment", "UnitsPurchased"], ascending=[True, False])
        .groupby("Segment")
        .first()
        .reset_index()
    )

    # Pareto analysis: Revenue and customer share
    total_rev = df_customers_segmented["Monetary"].sum()
    total_cust = len(df_customers_segmented)

    pareto_df = (
        df_customers_segmented.groupby("Segment")
        .agg(
            CustomerCount=("CustomerID", "count"),
            TotalRevenue=("Monetary", "sum")
        )
        .reset_index()
    )
    pareto_df["CustomerSharePct"] = (pareto_df["CustomerCount"] / total_cust) * 100.0
    pareto_df["RevenueSharePct"] = (pareto_df["TotalRevenue"] / total_rev) * 100.0
    pareto_df = pareto_df.sort_values("RevenueSharePct", ascending=False)

    return {
        "top_product_per_segment": top_product_by_segment,
        "pareto_summary": pareto_df
    }


def cluster_products(
    df_cleaned: pd.DataFrame,
    n_clusters: int = 4,
    save_csv: bool = True
) -> pd.DataFrame:
    """
    Clusters products based on commercial performance:
    - TotalUnits, TotalRevenue, AvgPrice (UnitPrice mean), DistinctCustomers
    Labels clusters intuitively: 'Best Sellers', 'High-Value / Premium', 'Steady Performers', 'Long-Tail'.
    """
    logger.info("Executing product catalog clustering...")
    p_agg = (
        df_cleaned.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            TotalUnits=("Quantity", "sum"),
            TotalRevenue=("Revenue", "sum"),
            AvgUnitPrice=("UnitPrice", "mean"),
            DistinctCustomers=("CustomerID", lambda x: x.dropna().nunique())
        )
    )

    features = ["TotalUnits", "TotalRevenue", "AvgUnitPrice", "DistinctCustomers"]
    X_p = np.log1p(p_agg[features].clip(lower=0))
    X_p_scaled = StandardScaler().fit_transform(X_p)

    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=RANDOM_STATE)
    p_agg["Cluster"] = km.fit_predict(X_p_scaled)

    # Profile clusters to assign meaningful domain labels
    cluster_stats = p_agg.groupby("Cluster").agg(
        MeanRevenue=("TotalRevenue", "mean"),
        MeanUnits=("TotalUnits", "mean"),
        MeanPrice=("AvgUnitPrice", "mean")
    ).reset_index()

    # Assign labels based on profile ranking
    # Best Sellers: Highest Units and Revenue
    # High-Value: High Price, moderate/low units
    # Long-tail: Lowest Revenue and Units
    cluster_stats["VolScore"] = cluster_stats["MeanRevenue"].rank() + cluster_stats["MeanUnits"].rank()
    best_seller_c = cluster_stats.sort_values("VolScore", ascending=False).iloc[0]["Cluster"]
    long_tail_c = cluster_stats.sort_values("VolScore", ascending=True).iloc[0]["Cluster"]
    
    remaining = [c for c in cluster_stats["Cluster"] if c not in [best_seller_c, long_tail_c]]
    # Highest price among remaining
    rem_stats = cluster_stats[cluster_stats["Cluster"].isin(remaining)].sort_values("MeanPrice", ascending=False)
    
    labels_map = {
        best_seller_c: "Best Sellers (High Volume)",
        long_tail_c: "Long-Tail (Low Traction)"
    }
    if len(remaining) >= 2:
        labels_map[rem_stats.iloc[0]["Cluster"]] = "Premium / High-Value"
        labels_map[rem_stats.iloc[1]["Cluster"]] = "Steady Volume Performers"
    elif len(remaining) == 1:
        labels_map[remaining[0]] = "Steady Performers"

    p_agg["ProductTier"] = p_agg["Cluster"].map(labels_map).fillna("Standard Product")

    if save_csv:
        p_agg.to_csv(PRODUCT_CLUSTERS_CSV, index=False)
        logger.info(f"Saved product clusters to {PRODUCT_CLUSTERS_CSV}")

    return p_agg


def run_market_basket_analysis(
    df_cleaned: pd.DataFrame,
    country: str = "United Kingdom",
    min_support: float = 0.015,
    min_lift: float = 1.2,
    top_n_items: int = 150,
    save_csv: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Performs Market Basket Analysis (Apriori + Association Rules):
    - Subsets to specified country (default UK) for performance and coherent consumer behavior.
    - Focuses on the top N most frequent items to avoid sparse matrix combinatorial explosion.
    - Calculates association rules (support, confidence, lift).
    - Returns rules dataframe and a lookup dictionary for 'frequently bought together'.
    """
    logger.info(f"Running Market Basket Analysis for {country} transactions...")
    df_sub = df_cleaned[df_cleaned["Country"] == country].copy()

    # Filter to top N items by frequency
    top_items = (
        df_sub["Description"]
        .value_counts()
        .head(top_n_items)
        .index.tolist()
    )
    df_sub = df_sub[df_sub["Description"].isin(top_items)]

    # Basket matrix (InvoiceNo x Description)
    basket = (
        df_sub.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    # Binary encoding
    basket_sets = (basket > 0).astype(bool)

    logger.info(f"Basket matrix shape: {basket_sets.shape}")

    # Frequent itemsets
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        logger.warning(f"No itemsets found with min_support={min_support}. Retrying with min_support=0.01")
        frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

    # Association rules
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        rules = rules.sort_values("lift", ascending=False)
        
        # Convert frozensets to clean strings for display and serialization
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    else:
        rules = pd.DataFrame(columns=["antecedents_str", "consequents_str", "support", "confidence", "lift"])

    # Lookup mapping: product -> list of recommended associated products
    lookup = {}
    if not rules.empty:
        for _, row in rules.iterrows():
            ant = row["antecedents_str"]
            con = row["consequents_str"]
            if ant not in lookup:
                lookup[ant] = []
            lookup[ant].append({
                "recommended_item": con,
                "confidence": round(row["confidence"], 3),
                "lift": round(row["lift"], 2),
                "support": round(row["support"], 4)
            })

    if save_csv and not rules.empty:
        rules.to_csv(MARKET_BASKET_RULES_CSV, index=False)
        logger.info(f"Saved association rules to {MARKET_BASKET_RULES_CSV}")

    return rules, lookup


def generate_business_recommendations() -> Dict[str, Dict[str, Any]]:
    """
    Automated actionable business strategies tailored per customer segment.
    """
    return {
        "Champions": {
            "profile": "Highest monetary spend, top order frequency, and recent purchases.",
            "strategy": "Reward & Retain (VIP Treatment)",
            "actions": [
                "Enroll in exclusive early-access programs for new product launches.",
                "Offer dedicated account management or concierge customer support.",
                "Solicit product feedback and advocate testimonials.",
                "Avoid aggressive discount promotions; focus on premium bundle exclusives."
            ],
            "churn_risk": "Low",
            "priority": "High Retention"
        },
        "Loyal": {
            "profile": "Consistent repeat purchasers with solid average order value and low recency.",
            "strategy": "Upsell & Community Building",
            "actions": [
                "Introduce tiered loyalty rewards (points-based incentives for next spending milestone).",
                "Cross-sell complementary categories based on association rule insights.",
                "Invite to community forums or VIP referral referral programs with mutual credits."
            ],
            "churn_risk": "Low to Moderate",
            "priority": "Value Expansion"
        },
        "Potential Loyalists": {
            "profile": "Recent buyers with moderate order frequency and standard basket sizes.",
            "strategy": "Nurture & Increase Purchase Frequency",
            "actions": [
                "Deploy automated post-purchase onboarding emails with related product recommendations.",
                "Provide limited-time discounts on second/third orders to establish purchasing habits.",
                "Offer free shipping thresholds to increase average basket sizes."
            ],
            "churn_risk": "Moderate",
            "priority": "Growth Acceleration"
        },
        "At-Risk": {
            "profile": "Previously high/moderate spenders whose recency has lapsed significantly (>90-180 days).",
            "strategy": "Win-Back & Reactivation",
            "actions": [
                "Trigger automated 'We Miss You' personalized email campaigns with bespoke discount codes.",
                "Conduct customer satisfaction pulse surveys to diagnose reasons for drop-off.",
                "Highlight new arrivals and bestselling improvements in their previously preferred categories."
            ],
            "churn_risk": "High",
            "priority": "Reactivation"
        },
        "Lost / Hibernating": {
            "profile": "Lowest monetary contribution, single-order history, and long-elapsed recency.",
            "strategy": "Low-Cost Re-engagement or Deprioritize",
            "actions": [
                "Include in high-funnel seasonal clearance promotions only to minimize marketing acquisition costs.",
                "Do not spend premium ad-retargeting budget; use scheduled batch email sends.",
                "Clean mailing lists after 12 months of inactivity to preserve domain sender reputation."
            ],
            "churn_risk": "Critical / Churned",
            "priority": "Cost Optimization"
        }
    }
