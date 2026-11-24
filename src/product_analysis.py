"""
Product sales and catalog performance analysis module.
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
from src.config import TOP_PRODUCTS_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def analyze_products(df: pd.DataFrame, save_csv: bool = True) -> Dict[str, Any]:
    """
    Performs comprehensive product-level analytics:
    - Per-product aggregates: Units, Revenue, Orders (distinct invoices), Customers (distinct).
    - Top product by Units ("Most Sold Product"), Revenue, and Orders.
    - Top 10 products for each metric.
    - Top product per country (top 10 countries by volume/revenue) and per month.
    - Temporal aggregations: monthly, weekly, day-of-week, hour-of-day.
    - Geographic revenue breakdown (Total & Non-UK split).
    """
    logger.info("Computing product-level statistics...")

    # Per-product summary
    # Note: StockCode might have a standardized Description from cleaning.py
    product_stats = (
        df.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            TotalUnits=("Quantity", "sum"),
            TotalRevenue=("Revenue", "sum"),
            DistinctOrders=("InvoiceNo", "nunique"),
            DistinctCustomers=("CustomerID", lambda x: x.dropna().nunique())
        )
    )

    # Sort candidates
    top_by_units = product_stats.sort_values("TotalUnits", ascending=False).iloc[0].to_dict()
    top_by_revenue = product_stats.sort_values("TotalRevenue", ascending=False).iloc[0].to_dict()
    top_by_orders = product_stats.sort_values("DistinctOrders", ascending=False).iloc[0].to_dict()

    logger.info(
        f"Most Sold Product (by Units): '{top_by_units['Description']}' (StockCode: {top_by_units['StockCode']}) "
        f"with {top_by_units['TotalUnits']:,} units sold."
    )
    logger.info(
        f"Top Product by Revenue: '{top_by_revenue['Description']}' (StockCode: {top_by_revenue['StockCode']}) "
        f"with £{top_by_revenue['TotalRevenue']:,.2f} revenue."
    )
    logger.info(
        f"Top Product by Orders: '{top_by_orders['Description']}' (StockCode: {top_by_orders['StockCode']}) "
        f"with {top_by_orders['DistinctOrders']:,} orders."
    )

    # Top 10 by each metric
    top10_units = product_stats.sort_values("TotalUnits", ascending=False).head(10)
    top10_revenue = product_stats.sort_values("TotalRevenue", ascending=False).head(10)
    top10_orders = product_stats.sort_values("DistinctOrders", ascending=False).head(10)

    # Top countries by revenue
    country_revenue = (
        df.groupby("Country")
        .agg(
            TotalRevenue=("Revenue", "sum"),
            TotalOrders=("InvoiceNo", "nunique"),
            TotalCustomers=("CustomerID", lambda x: x.dropna().nunique())
        )
        .sort_values("TotalRevenue", ascending=False)
        .reset_index()
    )

    top10_countries = country_revenue.head(10)["Country"].tolist()

    # Top product per top country
    top_product_per_country = []
    for c in top10_countries:
        country_df = df[df["Country"] == c]
        p_agg = (
            country_df.groupby(["StockCode", "Description"], as_index=False)
            .agg(TotalUnits=("Quantity", "sum"), TotalRevenue=("Revenue", "sum"))
            .sort_values("TotalUnits", ascending=False)
        )
        if not p_agg.empty:
            top_p = p_agg.iloc[0].to_dict()
            top_p["Country"] = c
            top_product_per_country.append(top_p)
    top_product_per_country_df = pd.DataFrame(top_product_per_country)

    # Temporal aggregations
    df_temp = df.copy()
    df_temp["YearMonth"] = df_temp["InvoiceDate"].dt.to_period("M").astype(str)
    df_temp["YearWeek"] = df_temp["InvoiceDate"].dt.to_period("W").astype(str)
    df_temp["DayOfWeek"] = df_temp["InvoiceDate"].dt.day_name()
    df_temp["Hour"] = df_temp["InvoiceDate"].dt.hour

    monthly_trend = (
        df_temp.groupby("YearMonth")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Units=("Quantity", "sum")
        )
        .reset_index()
    )

    weekly_trend = (
        df_temp.groupby("YearWeek")
        .agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"))
        .reset_index()
    )

    # Top product per month
    top_product_per_month = (
        df_temp.groupby(["YearMonth", "StockCode", "Description"], as_index=False)
        .agg(TotalUnits=("Quantity", "sum"), TotalRevenue=("Revenue", "sum"))
        .sort_values(["YearMonth", "TotalUnits"], ascending=[True, False])
        .groupby("YearMonth")
        .first()
        .reset_index()
    )

    # Day of week by Hour heatmap data
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_hour_matrix = (
        df_temp.groupby(["DayOfWeek", "Hour"])["Revenue"]
        .sum()
        .unstack(fill_value=0)
        .reindex([d for d in day_order if d in df_temp["DayOfWeek"].unique()])
    )

    # Save top products summary
    if save_csv:
        product_stats.sort_values("TotalUnits", ascending=False).to_csv(TOP_PRODUCTS_CSV, index=False)
        logger.info(f"Saved product summary to {TOP_PRODUCTS_CSV}")

    return {
        "product_stats": product_stats,
        "most_sold_by_units": top_by_units,
        "top_by_revenue": top_by_revenue,
        "top_by_orders": top_by_orders,
        "top10_units": top10_units,
        "top10_revenue": top10_revenue,
        "top10_orders": top10_orders,
        "country_revenue": country_revenue,
        "top_product_per_country": top_product_per_country_df,
        "top_product_per_month": top_product_per_month,
        "monthly_trend": monthly_trend,
        "weekly_trend": weekly_trend,
        "dow_hour_matrix": dow_hour_matrix
    }
