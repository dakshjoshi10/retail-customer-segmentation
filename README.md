# 🛍️ Retail Customer Segmentation & Sales Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dakshjoshi10-retail-customer-segmentati-appstreamlit-app-0fkzum.streamlit.app/)
*(Live Cloud Deployment: [Open Live Application](https://dakshjoshi10-retail-customer-segmentati-appstreamlit-app-0fkzum.streamlit.app/))*

An enterprise-grade customer analytics and machine learning solution built on the **Kaggle UK Online Retail dataset** (~540k transactions). This system implements automated data cleaning pipelines with **cancellation netting**, product catalog intelligence, customer behavioral feature engineering (RFM+), multi-algorithm unsupervised clustering (**K-Means**, **DBSCAN**, **Agglomerative Hierarchical**), Market Basket Analysis, and an interactive Streamlit dashboard.

*All monetary figures are denominated in British Pounds (£).*

---

## 🏆 Key Findings (At a Glance)

| Metric / Dimension | Value / Finding | Business Context |
| :--- | :--- | :--- |
| **Most Sold Product (Units)** | **`POPCORN HOLDER`** (`StockCode: 22197`) | **56,432 units sold** across **1,389 distinct orders** and **406 distinct customers**. |
| **Top Product by Revenue** | **`REGENCY CAKESTAND 3 TIER`** (`StockCode: 22423`) | **£165,378.99** total revenue generated across 1,949 orders and 872 customers. |
| **Top Product by Cart Orders** | **`WHITE HANGING HEART T-LIGHT HOLDER`** (`85123A`) | Appeared in **2,188 distinct orders** (highest penetration; 35,068 units sold). |
| **Active Customer Base** | **4,324 unique customers** | Filtered from 519,879 valid transactions with positive monetary contribution. |
| **Cancellation Netting Impact** | **£485,586.01 netted out** | 2,766 fully cancelled orders removed; 4,826 partially cancelled sales reduced. |
| **Pareto Concentration** | **15.9% Champions generate 64.6% Revenue** | Champions + Potential Loyalists account for **82.1%** of all revenue. |
| **Wholesale / VIP Outliers** | **131 DBSCAN Noise Accounts** | Mean spend **£17,202** (vs. £1,432 for core clusters; 12.0x higher). |
| **Market Basket Rules** | **1,530 Association Rules** | High-lift product bundles identified for automated cross-selling. |

---

## 🏗️ Project Architecture

```
PROJECT-1/
├── data/
│   ├── raw/                       # Original Kaggle data.csv (~43 MB)
│   └── processed/                 # Cleaned Parquet/CSV, segments, rules, comparisons
├── src/
│   ├── config.py                  # Global paths, constants, random seed, thresholds
│   ├── data_loader.py             # Robust raw data reader with ISO-8859-1 encoding
│   ├── cleaning.py                # Pipeline cleaning with cancellation netting logic
│   ├── product_analysis.py        # Product-level metrics, trends, geo analysis
│   ├── features.py                # RFM+ calculation, log1p transform, scaling
│   ├── clustering.py              # K-Means, DBSCAN, Agglomerative, PCA
│   ├── evaluation.py              # Silhouette, Davies-Bouldin, ARI matrix, profiling
│   ├── insights.py                # Segment affinities, catalog clustering, MBA rules
│   └── pipeline.py                # End-to-end unified execution pipeline
├── app/
│   └── streamlit_app.py           # 5-page interactive analytics dashboard
├── reports/
│   └── figures/                   # Saved evaluation plots (elbow, k-distance, dendrogram)
├── notebooks/
│   ├── 01_eda_and_modeling.ipynb  # Fully executed narrative walkthrough
│   └── build_notebook.py          # Programmatic notebook builder and test runner
├── tests/
│   ├── test_cleaning.py           # Pytest suite for cleaning & cancellation netting
│   ├── test_features.py           # Pytest suite for RFM and feature calculations
│   └── test_dashboard.py          # Automated verification of all 5 Streamlit pages
├── requirements.txt               # Pinned dependencies
├── README.md                      # Architecture documentation and insights
├── .gitignore                     # Git security ignore patterns
└── .env.example                   # Security template for Kaggle token
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Kaggle account credentials (for automated raw data download)

### 2. Environment Initialization
```bash
# Clone or navigate to the repository
cd PROJECT-1

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Kaggle API Authentication Setup
Kaggle credentials must **never** be hardcoded or checked into source control. You can configure authentication via either option:

#### Option A: Kaggle Environment Variable (Recommended)
```bash
export KAGGLE_API_TOKEN="<YOUR_KAGGLE_API_TOKEN>"
```

#### Option B: Kaggle Configuration Directory
Configure credentials in `~/.kaggle/` with appropriate permissions (`chmod 600`).

Verify authentication and download the dataset:
```bash
kaggle datasets list -s ecommerce
kaggle datasets download -d carrie1/ecommerce-data -p data/raw --unzip
```

---

## 🚀 Execution Guide

### Run End-to-End Pipeline
Execute the full data ingestion, cleaning, cancellation netting, feature engineering, clustering, evaluation, and export steps in one unified command:
```bash
python -m src.pipeline
```
*Outputs generated in `data/processed/` and `reports/figures/`.*

### Run Unit & Dashboard Tests
```bash
PYTHONPATH=. pytest tests/
```

### Launch Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Access the application locally at `http://localhost:8501`.

---

## 📊 Dashboard Modules (Pages Overview)

1. **Page 1: Overview**
   - **Hero Card**: Displays the **Most Sold Product** (`POPCORN HOLDER`, 56,432 units across 1,389 orders) with contextual comparison against revenue and order leaders.
   - **Executive KPIs**: Total revenue (£8.26M), orders (19.8k), customers (4,324), unique products (3,877), and Average Order Value (£417.15).
   - **Top 10 Products**: Dynamic horizontal bar chart toggleable between Units Sold, Total Revenue (£), and Distinct Orders.
   - **Temporal Trends & Heatmaps**: Monthly revenue line chart and a Day-of-Week vs. Hour-of-Day transactional volume heatmap.
   - **Geographic Breakdown**: Revenue distribution highlighting international non-UK markets (Netherlands, EIRE, Germany).

2. **Page 2: Product Analytics**
   - **Product Catalog Clustering**: 4 data-driven catalog tiers (*Best Sellers*, *Steady Performers*, *Long-Tail*, *Premium/High-Value*).
   - **Searchable Catalog**: Instant filtering by StockCode or Description.
   - **Geographic & Monthly Best Sellers**: Top product per top country and per calendar month.
   - **Frequently Bought Together**: Interactive Market Basket cross-sell recommender backed by 1,530 association rules.

3. **Page 3: Customer Segmentation**
   - **Live Algorithm Switcher**: Seamlessly toggle between **K-Means**, **DBSCAN**, and **Agglomerative Hierarchical** clustering.
   - **Interactive Live Tuning**: Sliders for $k$ (2–10), $\varepsilon$ (0.3–2.0), `min_samples` (3–25), and linkage methods (*ward*, *complete*, *average*).
   - **3D & 2D Visualizations**: 2D PCA feature projection and interactive 3D RFM scatter plot.
   - **Centroid Radar & Donut**: Multi-dimensional centroid radar fingerprint and customer cohort breakdown.
   - **Mathematical Diagnostic Plots**: Embedded publication-quality Elbow/Silhouette curve, DBSCAN k-distance knee curve, and Hierarchical Dendrogram.

4. **Page 4: Model Comparison**
   - **Metrics Benchmark**: Side-by-side comparison of Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Noise %, and Runtimes.
   - **Side-by-Side Projections**: Visual comparison of cluster boundaries on 2D PCA projections.
   - **Adjusted Rand Index (ARI) Matrix**: Quantitative cluster similarity heatmap across algorithms.
   - **Domain Recommendation**: Rigorous rationale explaining why K-Means is best suited for CRM operationalization while DBSCAN serves as an invaluable VIP detector.

5. **Page 5: 360° Customer Explorer**
   - **Single Customer Lookup**: Query any CustomerID to view segment assignment, RFM statistics, AOV, and basket size.
   - **Transaction History**: Top 10 most purchased items by the selected customer.
   - **Automated Recommendations**: Tailored strategic actions (e.g. VIP rewards for Champions, win-back discounts for At-Risk).
   - **Data Export**: One-click download button for `customers_with_segments.csv`.

---

## 🔬 Modeling & Technical Methodology

### 1. Data Cleaning Funnel & Cancellation Netting
| Step | Description | Rows / Metrics | Business Impact |
| :---: | :--- | :---: | :--- |
| 0 | Raw Ingestion | 541,909 records | Base Kaggle dataset. |
| 1 | Datetime Parsing | 541,909 records | Datetime conversions and validation. |
| 2 | Drop Exact Duplicates | 536,641 records | -5,268 exact duplicate rows. |
| 3 | Remove Non-Product StockCodes | 533,805 records | -2,836 service/fee items (POST, D, M, etc.). |
| 4a | Cancellation Netting (Removed) | **2,766 rows** | Original sales completely eliminated by cancellations. |
| 4b | Cancellation Netting (Reduced) | **4,826 rows** | Original sales quantity reduced by partial cancellations. |
| 4c | Cancellation Netting (Revenue) | **£485,586.01** | Total phantom revenue removed from data. |
| 5 | Valid Sales Filter | **519,879 records** | All negative rows and cancellations cleared. |
| 6 | Standardize Descriptions | 519,879 records | Modal description mapped per StockCode. |
| - | Customer Clustering Partition | **388,517 records** | Active customers with valid CustomerID (**4,324 unique**). |

### 2. Feature Transformation Strategy
- **Winsorization**: Capping metrics at the 99.5th percentile prevents extreme corporate bulk purchasers from distorting cluster centroids while preserving rank ordering.
- **Log1p Transformation**: Customer financial and frequency distributions exhibit severe positive skewness. Applying $\ln(1 + x)$ linearizes exponentially distributed metrics.
- **StandardScaler**: Centers each dimension to $\mu = 0, \sigma = 1$, preventing high-magnitude monetary figures from dominating Euclidean distance calculations.

### 3. Clustering Benchmark & Evaluation

| Algorithm | Clusters ($k$) | Noise % | Silhouette | Davies-Bouldin | Calinski-Harabasz | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($k=5$) [WINNER]** | **5** | **0.00%** | **0.2227** | **1.2415** | **1717.33** | **0.14s** |
| **Agglomerative (Ward, $k=5$)** | 5 | 0.00% | 0.1637 | 1.4728 | 1412.01 | 0.12s |
| **DBSCAN ($\varepsilon=1.0, \text{ms}=5$)** | 2 | 3.03% | 0.3799* | 1.1345* | 255.48* | 0.07s |

*\*DBSCAN metrics evaluated on core non-noise points.*

### 4. Segment Taxonomy (K-Means $k=5$)
1. **Champions (15.93% cust / 64.57% rev):** Recency: 15.4 days, Frequency: 13.6 orders, Spend: £7,739.01. Top product: `POPCORN HOLDER` (37,799 units).
2. **Potential Loyalists (18.36% cust / 17.52% rev):** Recency: 87.6 days, Frequency: 2.6 orders, Spend: £1,822.51. Top product: `WORLD WAR 2 GLIDERS ASSTD DESIGNS` (13,777 units).
3. **Loyal (23.50% cust / 11.14% rev):** Recency: 32.7 days, Frequency: 4.1 orders, Spend: £905.76. Top product: `ASSORTED COLOUR BIRD ORNAMENT` (4,276 units).
4. **At-Risk (26.87% cust / 5.42% rev):** Recency: 157.6 days, Frequency: 1.4 orders, Spend: £384.85. Top product: `ESSENTIAL BALM 3.5G TIN IN ENVELOPE` (5,586 units).
5. **Lost / Hibernating (15.33% cust / 1.35% rev):** Recency: 158.2 days, Frequency: 1.6 orders, Spend: £168.36. Top product: `ANTIQUE SILVER T-LIGHT GLASS` (1,052 units).

---

## ⚖️ Design Decisions & Trade-Offs

1. **Cancellation Netting Implementation:**
   - *Decision:* Matched cancellations to earlier sales with identical `CustomerID` and `StockCode` at the closest earlier date, deducting quantities or dropping rows before filtering out non-positive sales.
   - *Rationale:* Eliminates phantom transactions (e.g. `PAPER CRAFT , LITTLE BIRDIE`, 80,995 units cancelled minutes after order) and removes £485,586.01 in artificial revenue.

2. **Retaining Non-CustomerID Records for Sales Analytics:**
   - *Decision:* Cleaned transactions without `CustomerID` (~131k rows) are retained for catalog, revenue, and product-trend analysis, but filtered out when building the customer-level RFM matrix.
   - *Rationale:* Dropping guest purchases would understate catalog demand and skew market basket analysis.

3. **Algorithm Recommendation Selection:**
   - *Decision:* Recommend **K-Means ($k=5$)** as the operational production segmenter, paired with **DBSCAN** as a specialized VIP detector.
   - *Rationale:* K-Means provides clear operational cohorts for CRM workflows, while DBSCAN isolates high-value wholesale accounts averaging £17,202.

---

## 📜 Data Attribution & License
- **Data Attribution:** The dataset used in this project is the UCI Machine Learning Repository Online Retail dataset, sourced via Kaggle (`carrie1/ecommerce-data`), and used here strictly for academic, research, and educational demonstration purposes.
- **License:** Distributed under the MIT License. See `LICENSE` for details.

---

## 🔒 Security Compliance
- Zero API tokens or credentials hardcoded in codebase, logs, notebook outputs, or artifacts.
- `.gitignore` strictly protects local environments, data folders, credentials, and virtual environment directories.
