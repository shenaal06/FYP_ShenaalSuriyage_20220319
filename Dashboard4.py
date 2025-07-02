import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

# --- Set up page config ---
st.set_page_config(page_title="Pharma Inventory AI Dashboard", layout="wide")

# --- Set background image using HTML/CSS ---
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1588776814546-ec7c1a2f50c4?auto=format&fit=crop&w=1500&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Expiry Risk", "Stock-Out Risk", "Overstocked"],
        icons=["house", "exclamation-triangle", "battery-empty", "box-seam"],
        default_index=0,
    )

# --- Load data ---
try:
    expiry_df = pd.read_csv("expiry_predictions.csv")
    stockout_df = pd.read_csv("stockout_predictions_full.csv")
    overstock_df = pd.read_csv("overstock_clusters.csv")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# --- HOME ---
if selected == "Home":
    st.title("📦 Intelligent Inventory Management Dashboard")
    st.markdown("""
    Welcome to the Pharmaceutical Inventory Dashboard. This tool provides machine learning-based insights on:
    - 🔻 Expiry Risk of drugs
    - ⚠️ Stock-Out Risk based on safety levels
    - 📈 Overstocking of medicines
    Use the sidebar to explore each risk area and take proactive decisions.
    """)

# --- EXPIRY RISK ---
elif selected == "Expiry Risk":
    st.title("🔻 Expiry Risk Medicines")
    if "Predicted Loss" in expiry_df.columns and "Drug_Name_Label" in expiry_df.columns:
        top_expiry = expiry_df.sort_values(by="Predicted Loss", ascending=False).head(10)
        st.dataframe(top_expiry[["Drug_Name_Label", "Predicted Loss", "Expiry Risk"]], use_container_width=True)

        fig = px.bar(top_expiry, x="Drug_Name_Label", y="Predicted Loss",
                     title="💸 Top 10 Expiry Risk Medicines",
                     labels={"Drug_Name_Label": "Drug Name", "Predicted Loss": "Predicted Expiry Loss ($)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("❗ Required columns not found in expiry_predictions.csv")

# --- STOCK-OUT RISK ---
elif selected == "Stock-Out Risk":
    st.title("⚠️ Medicines at Stock-Out Risk")
    if "Stock-Out Risk" in stockout_df.columns and "Drug_Name_Label" in stockout_df.columns:
        stockout_risks = stockout_df[stockout_df["Stock-Out Risk"] == "Yes"]
        st.metric(label="Total Stock-Out Risk Medicines", value=len(stockout_risks))
        st.dataframe(stockout_risks[["Drug_Name_Label", "Left Stock", "Safety Stock", "Probability"]], use_container_width=True)

        fig2 = px.histogram(stockout_risks, x="Drug_Name_Label", y="Probability",
                            title="📉 Probability of Stock-Out by Drug",
                            labels={"Drug_Name_Label": "Drug", "Probability": "Risk Score"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("❗ Required columns not found in stockout_predictions_full.csv")

# --- OVERSTOCK ---
elif selected == "Overstocked":
    st.title("📈 Overstocked Medicines")
    if "Overstock_Cluster" in overstock_df.columns and "Drug_Name_Label" in overstock_df.columns:
        overstock_cluster = overstock_df[overstock_df["Overstock_Cluster"] == 2]
        st.metric(label="Total Overstocked Drugs", value=len(overstock_cluster))
        st.dataframe(overstock_cluster[["Drug_Name_Label", "Left Stock", "Usage (months)"]], use_container_width=True)

        fig3 = px.scatter(overstock_cluster, x="Usage (months)", y="Left Stock",
                          hover_data=["Drug_Name_Label"],
                          title="📦 Overstocked Drugs by Usage vs Stock",
                          labels={"Left Stock": "Stock (Units)", "Usage (months)": "Monthly Usage"})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("❗ Required columns not found in overstock_clusters.csv")

st.markdown("---")
st.caption("© 2025 Predictive Pharmaceutical Inventory | Powered by ML & Streamlit")