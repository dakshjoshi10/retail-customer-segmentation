"""
Interactive Retail Customer Segmentation & Sales Intelligence Dashboard.
Built with Streamlit and Plotly.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CLEANED_DATA_FILE,
    CLEANED_DATA_CSV,
    CUSTOMERS_SEGMENTED_CSV,
    MODEL_COMPARISON_CSV,
    TOP_PRODUCTS_CSV,
    PRODUCT_CLUSTERS_CSV,
    MARKET_BASKET_RULES_CSV,
    SEGMENT_PROFILES_CSV,
    DEPLOY_TRANSACTIONS_FILE,
    DEPLOY_CUSTOMERS_FILE,
    DEPLOY_MODEL_COMPARISON_FILE,
    DEPLOY_TOP_PRODUCTS_FILE,
    DEPLOY_PRODUCT_CLUSTERS_FILE,
    DEPLOY_MARKET_BASKET_RULES_FILE,
    DEPLOY_SEGMENT_PROFILES_FILE,
    FIGURES_DIR,
    FEATURE_COLUMNS,
    RANDOM_STATE
)
from src.clustering import assign_kmeans_segment_names
from src.evaluation import compute_clustering_metrics, compute_segment_profiles, compute_ari_matrix
from src.insights import generate_business_recommendations

# Page configuration
st.set_page_config(
    page_title="Retail Sales & Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #f0f2f6; }
    .hero-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- Cached Data Loaders ----------------
@st.cache_data
def load_all_data():
    """Loads all precomputed data files with caching, prioritizing deploy directory."""
    # Transactions table
    if DEPLOY_TRANSACTIONS_FILE.exists():
        df_clean = pd.read_parquet(DEPLOY_TRANSACTIONS_FILE)
    elif CLEANED_DATA_FILE.exists():
        df_clean = pd.read_parquet(CLEANED_DATA_FILE)
    elif CLEANED_DATA_CSV.exists():
        df_clean = pd.read_csv(CLEANED_DATA_CSV)
        df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    else:
        raise FileNotFoundError("Transactions dataset not found in data/deploy/ or data/processed/.")

    # Customers table with segment labels
    cust_path = DEPLOY_CUSTOMERS_FILE if DEPLOY_CUSTOMERS_FILE.exists() else CUSTOMERS_SEGMENTED_CSV
    df_customers = pd.read_csv(cust_path, dtype={"CustomerID": str})

    # Products table
    prod_path = DEPLOY_TOP_PRODUCTS_FILE if DEPLOY_TOP_PRODUCTS_FILE.exists() else TOP_PRODUCTS_CSV
    df_products = pd.read_csv(prod_path)

    # Product clusters
    cluster_path = DEPLOY_PRODUCT_CLUSTERS_FILE if DEPLOY_PRODUCT_CLUSTERS_FILE.exists() else PRODUCT_CLUSTERS_CSV
    df_product_clusters = pd.read_csv(cluster_path)

    # Model comparison
    comp_path = DEPLOY_MODEL_COMPARISON_FILE if DEPLOY_MODEL_COMPARISON_FILE.exists() else MODEL_COMPARISON_CSV
    df_model_comp = pd.read_csv(comp_path)
    
    # Association rules
    rules_path = DEPLOY_MARKET_BASKET_RULES_FILE if DEPLOY_MARKET_BASKET_RULES_FILE.exists() else MARKET_BASKET_RULES_CSV
    if rules_path.exists():
        df_rules = pd.read_csv(rules_path)
    else:
        df_rules = pd.DataFrame()

    return df_clean, df_customers, df_products, df_product_clusters, df_model_comp, df_rules


@st.cache_data
def get_scaled_customer_features(df_customers):
    """Computes log-transformed and scaled customer feature matrix for live clustering."""
    X_raw = df_customers[FEATURE_COLUMNS].copy()
    # Upper capping at 99.5%
    for col in FEATURE_COLUMNS:
        q = X_raw[col].quantile(0.995)
        X_raw[col] = np.clip(X_raw[col], 0, q)
    X_log = np.log1p(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    return X_scaled, X_pca


try:
    df_clean, df_customers, df_products, df_product_clusters, df_model_comp, df_rules = load_all_data()
    X_scaled, X_pca = get_scaled_customer_features(df_customers)
except Exception as e:
    st.error(f"Error loading pipeline datasets: {e}. Please ensure `python -m src.pipeline` has run successfully.")
    st.stop()


# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🛍️ Retail Intelligence")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Product Analytics", "Customer Segmentation", "Model Comparison", "Customer Explorer"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

min_date = df_clean["InvoiceDate"].min().date()
max_date = df_clean["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

all_countries = ["All Countries"] + sorted(df_clean["Country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Country Filter", all_countries, index=0)

# Apply global filters to transaction data
filtered_df = df_clean.copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    filtered_df = filtered_df[(filtered_df["InvoiceDate"] >= start_dt) & (filtered_df["InvoiceDate"] < end_dt)]

if selected_country != "All Countries":
    filtered_df = filtered_df[filtered_df["Country"] == selected_country]

st.sidebar.markdown("---")
st.sidebar.caption("Retail Segmentation & Sales Intelligence Engine")
st.sidebar.caption("Dataset: Kaggle UK Online Retail (~540k records)")


# Helper for graceful empty filter state
def check_empty(df):
    if df.empty:
        st.warning("⚠️ No records match the active filter criteria. Please broaden your date range or country selection.")
        st.stop()


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page == "Overview":
    check_empty(filtered_df)

    # Compute top sold product dynamically for the active filtered data
    prod_agg = (
        filtered_df.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            TotalUnits=("Quantity", "sum"),
            TotalRevenue=("Revenue", "sum"),
            DistinctOrders=("InvoiceNo", "nunique"),
            DistinctCustomers=("CustomerID", lambda x: x.dropna().nunique())
        )
    )
    most_sold_units = prod_agg.sort_values("TotalUnits", ascending=False).iloc[0]
    top_rev_item = prod_agg.sort_values("TotalRevenue", ascending=False).iloc[0]
    top_ord_item = prod_agg.sort_values("DistinctOrders", ascending=False).iloc[0]

    # HERO CARD: Most Sold Product
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9; margin-bottom: 4px;">
            🏆 Most Sold Product (Highest Units Sold)
        </div>
        <div style="font-size: 2.2rem; font-weight: 800; margin-bottom: 8px;">
            {most_sold_units['Description']}
        </div>
        <div style="font-size: 1.05rem; opacity: 0.95; margin-bottom: 16px;">
            StockCode: <b>{most_sold_units['StockCode']}</b> &nbsp;|&nbsp; 
            Units Sold: <b>{most_sold_units['TotalUnits']:,}</b> &nbsp;|&nbsp; 
            Total Revenue: <b>£{most_sold_units['TotalRevenue']:,.2f}</b> &nbsp;|&nbsp; 
            Orders: <b>{most_sold_units['DistinctOrders']:,}</b> &nbsp;|&nbsp; 
            Unique Customers: <b>{most_sold_units['DistinctCustomers']:,}</b>
        </div>
        <div style="font-size: 0.85rem; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 8px; display: inline-block;">
            💡 <b>Product Comparison Insight:</b> Top by revenue is <i>{top_rev_item['Description']}</i> (£{top_rev_item['TotalRevenue']:,.2f}) and top by orders is <i>{top_ord_item['Description']}</i> ({top_ord_item['DistinctOrders']:,} orders).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI ROW
    total_rev = filtered_df["Revenue"].sum()
    total_orders = filtered_df["InvoiceNo"].nunique()
    total_cust = filtered_df["CustomerID"].dropna().nunique()
    unique_prods = filtered_df["StockCode"].nunique()
    total_countries = filtered_df["Country"].nunique()
    aov = total_rev / total_orders if total_orders > 0 else 0

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Revenue</div><div class="metric-value">£{total_rev:,.0f}</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Orders</div><div class="metric-value">{total_orders:,}</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Customers</div><div class="metric-value">{total_cust:,}</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Products</div><div class="metric-value">{unique_prods:,}</div></div>', unsafe_allow_html=True)
    with kpi5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Countries</div><div class="metric-value">{total_countries:,}</div></div>', unsafe_allow_html=True)
    with kpi6:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Order Val</div><div class="metric-value">£{aov:,.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TOP 10 PRODUCTS BAR CHART WITH METRIC TOGGLE
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 Top 10 Products")
        metric_toggle = st.radio("Rank by metric:", ["Units Sold", "Revenue (£)", "Distinct Orders"], horizontal=True)
        
        if metric_toggle == "Units Sold":
            sort_col = "TotalUnits"
            label_text = "Units Sold"
            chart_color = "#3b82f6"
        elif metric_toggle == "Revenue (£)":
            sort_col = "TotalRevenue"
            label_text = "Revenue (£)"
            chart_color = "#10b981"
        else:
            sort_col = "DistinctOrders"
            label_text = "Distinct Orders"
            chart_color = "#8b5cf6"

        top10_df = prod_agg.sort_values(sort_col, ascending=True).tail(10)
        fig_bar = px.bar(
            top10_df,
            x=sort_col,
            y="Description",
            orientation="h",
            labels={sort_col: label_text, "Description": "Product"},
            title=f"Top 10 Products by {label_text}",
            color_discrete_sequence=[chart_color],
            text=sort_col
        )
        fig_bar.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), xaxis_title=label_text, yaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("📈 Monthly Revenue Trend")
        filtered_df["YearMonth"] = filtered_df["InvoiceDate"].dt.to_period("M").astype(str)
        monthly_df = filtered_df.groupby("YearMonth", as_index=False)["Revenue"].sum()
        
        fig_trend = px.line(
            monthly_df,
            x="YearMonth",
            y="Revenue",
            markers=True,
            title="Revenue Over Time (Monthly)",
            labels={"YearMonth": "Month", "Revenue": "Revenue (£)"},
            line_shape="spline",
            color_discrete_sequence=["#06b6d4"]
        )
        fig_trend.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

    # DAY-OF-WEEK BY HOUR HEATMAP & GEOGRAPHY
    col_heat, col_geo = st.columns([1, 1])

    with col_heat:
        st.subheader("🕒 Sales Timing: Day-of-Week by Hour")
        filtered_df["DayOfWeek"] = filtered_df["InvoiceDate"].dt.day_name()
        filtered_df["Hour"] = filtered_df["InvoiceDate"].dt.hour
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
        dow_hour = (
            filtered_df.groupby(["DayOfWeek", "Hour"])["Revenue"]
            .sum()
            .unstack(fill_value=0)
            .reindex([d for d in dow_order if d in filtered_df["DayOfWeek"].unique()])
        )
        fig_heat = px.imshow(
            dow_hour,
            labels=dict(x="Hour of Day", y="Day of Week", color="Revenue (£)"),
            x=dow_hour.columns,
            y=dow_hour.index,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Revenue Heatmap (Day vs Hour)"
        )
        fig_heat.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_geo:
        st.subheader("🌍 Revenue by Country (Non-UK Spotlight)")
        country_agg = (
            df_clean.groupby("Country", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
        )
        geo_mode = st.radio("View:", ["All Countries", "Excluding United Kingdom"], horizontal=True)
        if geo_mode == "Excluding United Kingdom":
            country_display = country_agg[country_agg["Country"] != "United Kingdom"].head(10)
        else:
            country_display = country_agg.head(10)

        fig_country = px.bar(
            country_display.sort_values("Revenue", ascending=True),
            x="Revenue",
            y="Country",
            orientation="h",
            color="Revenue",
            color_continuous_scale="Viridis",
            title=f"Top 10 Markets by Revenue ({geo_mode})"
        )
        fig_country.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="")
        st.plotly_chart(fig_country, use_container_width=True)


# ==============================================================================
# PAGE 2: PRODUCTS
# ==============================================================================
elif page == "Product Analytics":
    st.title("📦 Product Catalog Intelligence")
    check_empty(filtered_df)

    tab1, tab2, tab3 = st.tabs(["Catalog Explorer & Tiers", "Geographic & Temporal Leaders", "Market Basket / Cross-Sell"])

    with tab1:
        st.subheader("Product Catalog Search & Tiering")
        
        # Product tier distribution
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            tier_counts = df_product_clusters["ProductTier"].value_counts().reset_index()
            tier_counts.columns = ["ProductTier", "Count"]
            fig_pie = px.pie(
                tier_counts,
                names="ProductTier",
                values="Count",
                hole=0.45,
                title="Product Catalog Segmentation",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_t2:
            fig_scatter = px.scatter(
                df_product_clusters,
                x="TotalUnits",
                y="TotalRevenue",
                color="ProductTier",
                hover_data=["Description", "AvgUnitPrice", "DistinctCustomers"],
                log_x=True,
                log_y=True,
                title="Product Clusters: Total Units vs Revenue (Log-Log Scale)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Searchable product table
        search_query = st.text_input("🔍 Search products by Description or StockCode:", "")
        filtered_products = df_product_clusters.copy()
        if search_query:
            filtered_products = filtered_products[
                filtered_products["Description"].str.contains(search_query, case=False, na=False) |
                filtered_products["StockCode"].str.contains(search_query, case=False, na=False)
            ]
        st.dataframe(
            filtered_products[["StockCode", "Description", "TotalUnits", "TotalRevenue", "AvgUnitPrice", "DistinctCustomers", "ProductTier"]]
            .sort_values("TotalUnits", ascending=False),
            use_container_width=True,
            height=350
        )

    with tab2:
        st.subheader("Geographic & Monthly Best Sellers")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### Top Product in Top 10 Countries")
            country_top_products = []
            top_countries = df_clean.groupby("Country")["Revenue"].sum().nlargest(10).index
            for c in top_countries:
                c_df = df_clean[df_clean["Country"] == c]
                p = (
                    c_df.groupby(["StockCode", "Description"], as_index=False)["Quantity"]
                    .sum()
                    .sort_values("Quantity", ascending=False)
                    .iloc[0]
                )
                country_top_products.append({"Country": c, "StockCode": p["StockCode"], "Top Product": p["Description"], "Units": p["Quantity"]})
            st.dataframe(pd.DataFrame(country_top_products), use_container_width=True)

        with col_g2:
            st.markdown("#### Top Product per Month")
            df_clean_copy = df_clean.copy()
            df_clean_copy["YearMonth"] = df_clean_copy["InvoiceDate"].dt.to_period("M").astype(str)
            month_top = (
                df_clean_copy.groupby(["YearMonth", "StockCode", "Description"], as_index=False)["Quantity"]
                .sum()
                .sort_values(["YearMonth", "Quantity"], ascending=[True, False])
                .groupby("YearMonth")
                .first()
                .reset_index()
            )
            st.dataframe(month_top[["YearMonth", "StockCode", "Description", "Quantity"]], use_container_width=True)

    with tab3:
        st.subheader("🛒 Frequently Bought Together (Market Basket Analysis)")
        st.markdown("Discover actionable product association rules discovered from high-volume transaction carts.")

        if not df_rules.empty:
            antecedents_list = sorted(df_rules["antecedents_str"].unique().tolist())
            selected_item = st.selectbox("Select or type a product to find recommendations:", antecedents_list)

            item_rules = df_rules[df_rules["antecedents_str"] == selected_item].sort_values("lift", ascending=False)
            if not item_rules.empty:
                st.markdown(f"#### Customers who bought **{selected_item}** also frequently purchased:")
                for _, r in item_rules.head(5).iterrows():
                    st.info(f"✨ **{r['consequents_str']}** (Lift: **{r['lift']:.2f}x**, Confidence: **{r['confidence'] * 100:.1f}%**, Cart Support: **{r['support'] * 100:.2f}%**)")
            else:
                st.info("No high-lift association rules found for this individual item.")

            st.markdown("---")
            st.markdown("#### Top 15 Overall Association Rules (by Lift)")
            st.dataframe(
                df_rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]]
                .rename(columns={"antecedents_str": "Antecedents (If bought)", "consequents_str": "Consequents (Also bought)", "lift": "Lift Multiplier"})
                .head(15),
                use_container_width=True
            )
        else:
            st.info("Market basket rules table not loaded or empty.")


# ==============================================================================
# PAGE 3: CUSTOMER SEGMENTATION
# ==============================================================================
elif page == "Customer Segmentation":
    st.title("🎯 Customer Segmentation Engine")
    st.markdown("Interactive multi-algorithm clustering with live hyperparameter tuning.")

    # Model selector and live parameters
    col_model, col_params = st.columns([1, 2])
    with col_model:
        selected_algo = st.selectbox(
            "Select Clustering Algorithm:",
            ["K-Means", "DBSCAN", "Agglomerative Hierarchical"]
        )

    # Dynamic live parameter controls
    with col_params:
        if selected_algo == "K-Means":
            k_val = st.slider("Number of Clusters (k):", min_value=2, max_value=10, value=5, step=1)
            # Live fit
            km = KMeans(n_clusters=k_val, init="k-means++", n_init=20, random_state=RANDOM_STATE)
            live_labels = km.fit_predict(X_scaled)
            df_cust_live = assign_kmeans_segment_names(df_customers, live_labels)
            label_col = "Segment"

        elif selected_algo == "DBSCAN":
            c_eps, c_ms = st.columns(2)
            with c_eps:
                eps_val = st.slider("Epsilon (Neighborhood radius):", min_value=0.3, max_value=2.0, value=1.0, step=0.1)
            with c_ms:
                ms_val = st.slider("Min Samples (Density threshold):", min_value=3, max_value=25, value=5, step=1)
            db = DBSCAN(eps=eps_val, min_samples=ms_val)
            live_labels = db.fit_predict(X_scaled)
            df_cust_live = df_customers.copy()
            df_cust_live["Segment"] = [f"Cluster {l}" if l != -1 else "Noise (VIP / Outlier)" for l in live_labels]
            label_col = "Segment"

        else: # Agglomerative
            c_k, c_link = st.columns(2)
            with c_k:
                agg_k = st.slider("Number of Clusters (k):", min_value=2, max_value=10, value=5, step=1)
            with c_link:
                agg_link = st.selectbox("Linkage Criterion:", ["ward", "complete", "average"])
            agg = AgglomerativeClustering(n_clusters=agg_k, linkage=agg_link)
            live_labels = agg.fit_predict(X_scaled)
            df_cust_live = df_customers.copy()
            df_cust_live["Segment"] = [f"Cluster {l}" for l in live_labels]
            label_col = "Segment"

    # Add PCA coordinates to live dataframe
    df_cust_live["PCA1"] = X_pca[:, 0]
    df_cust_live["PCA2"] = X_pca[:, 1]

    # Quick metrics banner for current active clustering
    n_unique_clusters = len(set(live_labels)) - (1 if -1 in live_labels else 0)
    noise_count = int(np.sum(live_labels == -1))
    noise_pct = (noise_count / len(live_labels)) * 100.0

    st.markdown(f"""
    <div style="background:#1e293b; padding:12px 18px; border-radius:8px; margin-bottom:18px;">
        ⚡ <b>Active Model:</b> {selected_algo} &nbsp;|&nbsp; 
        <b>Clusters Formed:</b> {n_unique_clusters} &nbsp;|&nbsp; 
        <b>Noise Points:</b> {noise_count} ({noise_pct:.1f}%) &nbsp;|&nbsp;
        <b>Total Customers:</b> {len(df_cust_live):,}
    </div>
    """, unsafe_allow_html=True)

    # VISUALIZATION ROW 1: 2D PCA & 3D RFM SCATTER
    col_scat2d, col_scat3d = st.columns(2)
    with col_scat2d:
        st.subheader("2D PCA Customer Projection")
        fig_pca = px.scatter(
            df_cust_live,
            x="PCA1",
            y="PCA2",
            color=label_col,
            hover_data=["CustomerID", "Monetary", "Frequency", "Recency"],
            title=f"2D PCA Feature Space ({selected_algo})",
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.75
        )
        fig_pca.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_scat3d:
        st.subheader("3D RFM Feature Space")
        fig_3d = px.scatter_3d(
            df_cust_live,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color=label_col,
            log_y=True,
            log_z=True,
            hover_data=["CustomerID"],
            title="3D RFM Space (Log Scale Frequency & Monetary)",
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.7
        )
        fig_3d.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_3d, use_container_width=True)

    # VISUALIZATION ROW 2: DONUT CHART & REVENUE SHARE (PARETO)
    col_donut, col_pareto = st.columns(2)
    with col_donut:
        st.subheader("Customer Distribution by Segment")
        seg_counts = df_cust_live[label_col].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_donut = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_donut.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_pareto:
        st.subheader("Revenue Share per Segment (Pareto View)")
        seg_rev = df_cust_live.groupby(label_col)["Monetary"].sum().reset_index()
        seg_rev.columns = ["Segment", "TotalRevenue"]
        seg_rev["RevenuePct"] = (seg_rev["TotalRevenue"] / seg_rev["TotalRevenue"].sum()) * 100
        fig_rev = px.bar(
            seg_rev.sort_values("RevenuePct", ascending=False),
            x="Segment",
            y="RevenuePct",
            color="Segment",
            text=seg_rev["RevenuePct"].apply(lambda x: f"{x:.1f}%"),
            title="Share of Total Revenue (%)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_rev.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Revenue %")
        st.plotly_chart(fig_rev, use_container_width=True)

    # PROFILE TABLE & CENTROID RADAR
    st.subheader("📋 Segment Profiles & Empirical Centroids")
    profile_df = compute_segment_profiles(df_cust_live, cluster_col=label_col)
    st.dataframe(profile_df, use_container_width=True)

    # Radar chart for normalized centroid metrics
    st.subheader("🕸️ Centroid Fingerprint (Radar Chart)")
    radar_metrics = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]
    # Normalize 0..1 for radar
    radar_data = df_cust_live.groupby(label_col)[radar_metrics].mean()
    radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-6)

    fig_radar = go.Figure()
    for seg in radar_norm.index:
        values = radar_norm.loc[seg].tolist()
        values += [values[0]]  # close loop
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself",
            name=str(seg),
            opacity=0.6
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # DIAGNOSTIC EVALUATION PLOTS (Elbow, k-distance, Dendrogram)
    st.subheader("🔬 Mathematical Diagnostics & Validation Plots")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("#### K-Means Elbow & Silhouette")
        p1 = FIGURES_DIR / "kmeans_elbow_silhouette.png"
        if p1.exists():
            st.image(str(p1), caption="Inertia Elbow and Silhouette across k=2..10")
    with col_d2:
        st.markdown("#### DBSCAN k-Distance Graph")
        p2 = FIGURES_DIR / "dbscan_kdistance_plot.png"
        if p2.exists():
            st.image(str(p2), caption="Knee detection for optimal epsilon")
    with col_d3:
        st.markdown("#### Hierarchical Dendrogram")
        p3 = FIGURES_DIR / "hierarchical_dendrogram.png"
        if p3.exists():
            st.image(str(p3), caption="Ward linkage hierarchical tree")


# ==============================================================================
# PAGE 4: MODEL COMPARISON
# ==============================================================================
elif page == "Model Comparison":
    st.title("⚖️ Comprehensive Clustering Model Comparison")
    st.markdown("Direct objective comparison between K-Means, DBSCAN, and Agglomerative Hierarchical Clustering.")

    # Metrics Table
    st.subheader("📊 Performance & Separation Benchmark")
    st.dataframe(df_model_comp, use_container_width=True)

    # Side-by-side cluster projections
    st.subheader("🗺️ Side-by-Side 2D Feature Projections")
    col_k, col_db, col_ag = st.columns(3)
    
    with col_k:
        fig_km_side = px.scatter(
            df_customers,
            x="PCA1",
            y="PCA2",
            color="Segment",
            title="K-Means (k=5)",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_km_side.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_km_side, use_container_width=True)

    with col_db:
        df_customers["DBSCAN_Group"] = df_customers["DBSCAN_Cluster"].apply(lambda x: "Noise / VIP" if x == -1 else f"Cluster {x}")
        fig_db_side = px.scatter(
            df_customers,
            x="PCA1",
            y="PCA2",
            color="DBSCAN_Group",
            title="DBSCAN Density Partition",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_db_side.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_db_side, use_container_width=True)

    with col_ag:
        df_customers["Agg_Group"] = df_customers["Agglomerative_Cluster"].apply(lambda x: f"Cluster {x}")
        fig_ag_side = px.scatter(
            df_customers,
            x="PCA1",
            y="PCA2",
            color="Agg_Group",
            title="Agglomerative Ward (k=5)",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_ag_side.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_ag_side, use_container_width=True)

    # Adjusted Rand Index Matrix
    st.subheader("🔄 Adjusted Rand Index (ARI) Cross-Model Agreement")
    ari_mat = compute_ari_matrix({
        "K-Means": df_customers["KMeans_Cluster"].values,
        "DBSCAN": df_customers["DBSCAN_Cluster"].values,
        "Agglomerative": df_customers["Agglomerative_Cluster"].values
    })
    fig_ari = px.imshow(
        ari_mat,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Adjusted Rand Index Heatmap (Similarity: 0=Independent, 1=Identical)"
    )
    fig_ari.update_layout(height=360, width=500)
    st.plotly_chart(fig_ari, use_container_width=True)

    # Written Conclusion
    st.subheader("💡 Expert Algorithm Conclusion & Selection Rationale")
    st.markdown("""
    ### Why **K-Means (k=5)** is the Best Overall Fit for this Retail Environment:
    
    1. **Actionable Segmentation Structure:**
       - Retail marketing and CRM teams require distinct, exhaustive operational tiers (e.g. *Champions*, *Loyal*, *Potential Loyalists*, *At-Risk*, *Lost*).
       - K-Means assigns every customer cleanly into an interpretable cohort with clear gradient centroids across Recency, Frequency, and Monetary dimensions.
       
    2. **Superior Internal Separation Metrics:**
       - K-Means achieved the highest Calinski-Harabasz score (**1735.8**) and solid Silhouette score (**0.224**), outperforming Agglomerative Ward (**1427.2**) while taking less than a tenth of a second to compute.
       
    3. **The Vital Complementary Role of DBSCAN:**
       - While DBSCAN is less suitable as the primary marketing segmenter due to collapsing high-density customer regions into one dominant cluster, its density-based noise filtering is invaluable as an **Outlier / VIP Detector**.
       - DBSCAN isolated **135 extreme outlier accounts** whose average spending was **$19,764** (over 13x higher than core retail customers), capturing institutional wholesale accounts and VIP corporate buyers for dedicated B2B account management.
    """)


# ==============================================================================
# PAGE 5: CUSTOMER EXPLORER
# ==============================================================================
elif page == "Customer Explorer":
    st.title("👤 360° Individual Customer Explorer")
    st.markdown("Lookup individual customer profiles, RFM behavior, favorite products, and history.")

    col_input, col_meta = st.columns([1, 2])
    with col_input:
        available_ids = sorted(df_customers["CustomerID"].astype(str).unique().tolist())
        cust_id_input = st.text_input("Enter CustomerID (e.g. 17850, 14646, 18102):", value=available_ids[0])

    if cust_id_input not in available_ids:
        st.error(f"CustomerID '{cust_id_input}' not found in segmented customer dataset. Please check the ID.")
    else:
        cust_record = df_customers[df_customers["CustomerID"] == cust_id_input].iloc[0]
        
        # Segment and details
        st.markdown(f"""
        <div style="background:#1e293b; padding:18px; border-radius:10px; margin-bottom:20px; border-left:6px solid #3b82f6;">
            <h3 style="margin:0 0 8px 0;">Customer ID: {cust_record['CustomerID']} ({cust_record['Country']})</h3>
            <span class="badge" style="background:#3b82f6; font-size:1rem;">Segment: {cust_record['Segment']}</span>
            <span class="badge" style="background:#10b981; font-size:1rem;">Total Spend: £{cust_record['Monetary']:,.2f}</span>
            <span class="badge" style="background:#8b5cf6; font-size:1rem;">Orders: {int(cust_record['Frequency'])}</span>
            <span class="badge" style="background:#f59e0b; font-size:1rem;">Recency: {int(cust_record['Recency'])} days ago</span>
        </div>
        """, unsafe_allow_html=True)

        # Behavioral metric cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Avg Order Value", f"£{cust_record['AvgOrderValue']:,.2f}")
        with m2:
            st.metric("Unique Products", f"{int(cust_record['UniqueProducts'])}")
        with m3:
            st.metric("Avg Basket Size", f"{cust_record['AvgBasketSize']:.1f} units")
        with m4:
            st.metric("K-Means Cluster", f"{int(cust_record['KMeans_Cluster'])}")
        with m5:
            st.metric("DBSCAN Status", "VIP / Noise" if cust_record['DBSCAN_Cluster'] == -1 else f"Cluster {int(cust_record['DBSCAN_Cluster'])}")

        st.markdown("---")
        col_hist, col_recs = st.columns([1, 1])

        with col_hist:
            st.subheader("🛍️ Customer Purchase History & Top Products")
            cust_tx = df_clean[df_clean["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True) == cust_id_input]
            if not cust_tx.empty:
                cust_top_items = (
                    cust_tx.groupby(["StockCode", "Description"], as_index=False)
                    .agg(TotalQuantity=("Quantity", "sum"), TotalSpent=("Revenue", "sum"))
                    .sort_values("TotalQuantity", ascending=False)
                    .head(10)
                )
                st.dataframe(cust_top_items, use_container_width=True)
            else:
                st.info("No transaction history records found for this ID in cleaned dataset.")

        with col_recs:
            st.subheader("🎯 Tailored Business Recommendations")
            recs_dict = generate_business_recommendations()
            seg_rec = recs_dict.get(cust_record["Segment"], recs_dict["Champions"])

            st.markdown(f"**Strategy:** `{seg_rec['strategy']}`")
            st.markdown(f"**Retention Priority:** `{seg_rec['priority']}` &nbsp;|&nbsp; **Churn Risk:** `{seg_rec['churn_risk']}`")
            st.markdown("**Recommended Actions:**")
            for action in seg_rec["actions"]:
                st.markdown(f"- {action}")

    # Export Download Button
    st.markdown("---")
    st.subheader("📥 Export Segmented Customer Intelligence")
    csv_bytes = df_customers.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Segmented Customers CSV",
        data=csv_bytes,
        file_name="customers_with_segments.csv",
        mime="text/csv",
        help="Export complete customer list with RFM metrics, PCA coordinates, and segment labels."
    )
