"""
Customer 360 - Interactive Streamlit Dashboard
Acts as the BI presentation layer for the Customer 360 data pipeline.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Customer 360 Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DATA LOADING & ERROR HANDLING
# ==========================================
@st.cache_data
def load_data():
    """Loads pipeline outputs with graceful error handling if files are missing."""
    # Assuming script is in src/ folder, resolving to project root
    root_dir = Path(__file__).resolve().parents[1]
    outputs_dir = root_dir / "outputs"
    
    try:
        c360 = pd.read_csv(outputs_dir / "customer_360.csv")
        kpi_summary = pd.read_csv(outputs_dir / "kpi_summary.csv")
        cat_rev = pd.read_csv(outputs_dir / "category_revenue.csv")
        return c360, kpi_summary, cat_rev
    except FileNotFoundError:
        st.error("⚠️ **Data Assets Not Found!**")
        st.warning(
            "It looks like the outputs folder is missing or empty. "
            "Please run the data engine (`python pipeline.py`) in your terminal first to generate the datasets."
        )
        st.stop()  # Halt execution gracefully

# Load datasets
df_c360, df_kpi, df_cat_rev = load_data()


# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3126/3126647.png", width=60)
st.sidebar.title("Data Filters")
st.sidebar.markdown("Use the dropdowns below to slice the Customer 360 data.")

# Dynamic filter options based on the dataset
regions = df_c360["region"].dropna().unique().tolist()
segments = df_c360["segment"].dropna().unique().tolist()
industries = df_c360["industry"].dropna().unique().tolist()
tiers = df_c360["value_tier"].dropna().unique().tolist()

selected_regions = st.sidebar.multiselect("🌍 Region", options=regions, default=regions)
selected_segments = st.sidebar.multiselect("🏢 Segment", options=segments, default=segments)
selected_industries = st.sidebar.multiselect("🏭 Industry", options=industries, default=industries)
selected_tiers = st.sidebar.multiselect("💎 Value Tier", options=tiers, default=tiers)

# Apply filters
filtered_df = df_c360[
    (df_c360["region"].isin(selected_regions)) &
    (df_c360["segment"].isin(selected_segments)) &
    (df_c360["industry"].isin(selected_industries)) &
    (df_c360["value_tier"].isin(selected_tiers))
]


# ==========================================
# MAIN DASHBOARD UI
# ==========================================
st.title("🌐 Customer 360 Analytics Dashboard")
st.markdown("Analyze customer demographics, support health, and revenue performance.")
st.divider()

# --- TOP-LEVEL KPI CARDS ---
# We calculate these dynamically based on the applied filters for a highly interactive UX
total_customers = len(filtered_df)
total_revenue = filtered_df["total_net_revenue"].sum()
total_orders = filtered_df["order_count"].sum()

# Safely compute averages to avoid division by zero
aov = (total_revenue / total_orders) if total_orders > 0 else 0
avg_sat_score = filtered_df["average_satisfaction_score"].mean()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="👥 Total Customers", value=f"{total_customers:,}")
with col2:
    st.metric(label="💰 Total Net Revenue", value=f"${total_revenue:,.2f}")
with col3:
    st.metric(label="🛒 Average Order Value (AOV)", value=f"${aov:,.2f}")
with col4:
    st.metric(label="⭐ Avg Satisfaction Score", value=f"{avg_sat_score:.1f} / 5.0")

st.write("") # Spacer

# --- INTERACTIVE CHARTS ---
st.subheader("📊 Revenue Analytics")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Bar Chart: Revenue by Region (Dynamic based on filters)
    region_agg = filtered_df.groupby("region", as_index=False)["total_net_revenue"].sum()
    fig_bar = px.bar(
        region_agg, 
        x="region", 
        y="total_net_revenue", 
        title="Net Revenue by Region (Filtered)",
        labels={"region": "Region", "total_net_revenue": "Net Revenue ($)"},
        color="region",
        template="plotly_white"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    # Pie Chart: Revenue by Product Category (Global from category_revenue.csv)
    fig_pie = px.pie(
        df_cat_rev, 
        names="product_category", 
        values="total_net_revenue", 
        title="Global Revenue by Product Category",
        hole=0.4, # Makes it a donut chart for better aesthetics
        template="plotly_white"
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# --- HIGH-RISK CUSTOMER VIEW ---
st.divider()
st.subheader("🚨 High-Risk Customers")
st.markdown("Customers flagged with low satisfaction (< 2.5) OR high ticket volume (> 4) combined with slow resolution times (> 48h).")

risk_df = filtered_df[filtered_df["risk_flag"] == 1]

if not risk_df.empty:
    # Highlight the risk_flag column in red
    st.dataframe(
        risk_df.style.map(lambda x: "background-color: #ffcccc; color: #990000", subset=["risk_flag"]),
        use_container_width=True,
        hide_index=True
    )
else:
    st.success("✅ Excellent! There are no high-risk customers in the current filtered view.")


# --- INTERACTIVE CUSTOMER DETAIL TABLE & EXPORT ---
st.divider()
st.subheader("📑 Customer Detail Directory")

# Display the main filtered dataframe
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# Data Export Module
st.write("Need to share this data or analyze it in Excel?")
csv_export = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=csv_export,
    file_name="filtered_customer_360_export.csv",
    mime="text/csv",
    help="Click here to download the current table based on your sidebar filters."
)
