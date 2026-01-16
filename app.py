import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Streamlit Settings ----------------
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# ✅ Public dataset URL (Online Retail - UCI style hosted on GitHub raw)
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/online-retail.csv"

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    kmeans = pickle.load(open("models/kmeans_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    return kmeans, scaler

kmeans, scaler = load_models()

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="latin1")
    return df

# ---------------- Build Similarity Matrix ----------------
@st.cache_resource
def build_similarity_matrix(df):
    # Minimum cleaning for recommender
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["Description"] = df["Description"].astype(str).str.strip()
    df["CustomerID"] = df["CustomerID"].astype(int)

    pivot = df.pivot_table(
        index="Description",
        columns="CustomerID",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    similarity = cosine_similarity(pivot)
    similarity_df = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)
    return similarity_df

# ---------------- Recommendation Function ----------------
def recommend_products(product_name, similarity_df, top_n=5):
    if product_name is None or product_name.strip() == "":
        return []

    product_name_clean = product_name.strip().upper()
    product_map = {p.upper(): p for p in similarity_df.index}

    if product_name_clean not in product_map:
        return None

    actual_name = product_map[product_name_clean]
    scores = similarity_df[actual_name].sort_values(ascending=False)
    return scores.iloc[1:top_n+1].index.tolist()

# ---------------- Segment Mapping ----------------
final_segment_map = {
    2: "High-Value 💎",
    3: "Regular 🙂",
    0: "Occasional 🛒",
    1: "At-Risk ⚠️"
}

# ---------------- UI ----------------
st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation (RFM + KMeans) + Product Recommendations (Item-Based CF)")

with st.spinner("Loading dataset & building similarity matrix (first run may take time)..."):
    df = load_data()
    similarity_df = build_similarity_matrix(df)

tab1, tab2 = st.tabs(["🛍 Product Recommendation", "🎯 Customer Segmentation"])

# ---------------- TAB 1: Recommendation ----------------
with tab1:
    st.subheader("🛍 Item-Based Product Recommendations")

    product_name = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        recs = recommend_products(product_name, similarity_df)

        if recs is None:
            st.error("❌ Product not found. Try a different name.")
        elif len(recs) == 0:
            st.warning("⚠️ Please enter a product name.")
        else:
            st.success("✅ Top 5 Similar Products:")
            for i, p in enumerate(recs, start=1):
                st.write(f"**{i}. {p}**")

# ---------------- TAB 2: Segmentation ----------------
with tab2:
    st.subheader("🎯 Customer Segmentation Prediction")

    recency = st.number_input("Recency (days)", min_value=0, value=30)
    frequency = st.number_input("Frequency", min_value=0, value=5)
    monetary = st.number_input("Monetary", min_value=0.0, value=1000.0)

    if st.button("Predict Cluster"):
        user_data = np.array([[recency, frequency, monetary]])
        user_scaled = scaler.transform(user_data)

        cluster = kmeans.predict(user_scaled)[0]
        segment = final_segment_map.get(cluster, "Unknown Segment")

        st.success(f"✅ Predicted Segment: **{segment}**")
        st.info(f"Cluster ID: {cluster}")
