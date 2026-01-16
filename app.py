import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- Load Models ----------------
kmeans = pickle.load(open("models/kmeans_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
similarity_df = pickle.load(open("models/product_similarity.pkl", "rb"))

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
    recommendations = scores.iloc[1:top_n+1].index.tolist()
    return recommendations

# ---------------- Segment Mapping (Your Final Clusters) ----------------
final_segment_map = {
    2: "High-Value 💎",
    3: "Regular 🙂",
    0: "Occasional 🛒",
    1: "At-Risk ⚠️"
}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation (RFM + KMeans) + Product Recommendations (Item-Based Collaborative Filtering)")

tab1, tab2 = st.tabs(["🛍 Product Recommendation", "🎯 Customer Segmentation"])


# =====================================================
# TAB 1: Product Recommendation
# =====================================================
with tab1:
    st.subheader("🛍 Item-Based Product Recommendations")

    product_name = st.text_input("Enter Product Name (Exact or close match)")

    if st.button("Get Recommendations"):
        recs = recommend_products(product_name, similarity_df)

        if recs is None:
            st.error("❌ Product not found. Please check spelling or try another product.")
        elif len(recs) == 0:
            st.warning("⚠️ Please enter a product name.")
        else:
            st.success("✅ Top 5 Similar Products:")
            for i, p in enumerate(recs, start=1):
                st.write(f"**{i}. {p}**")


# =====================================================
# TAB 2: Customer Segmentation
# =====================================================
with tab2:
    st.subheader("🎯 Customer Segmentation Prediction")

    st.write("Enter RFM values to predict customer segment:")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of invoices)", min_value=0, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=1000.0)

    if st.button("Predict Cluster"):
        user_data = np.array([[recency, frequency, monetary]])
        user_scaled = scaler.transform(user_data)

        cluster = kmeans.predict(user_scaled)[0]
        segment = final_segment_map.get(cluster, "Unknown Segment")

        st.success(f"✅ Predicted Segment: **{segment}**")
        st.info(f"Cluster ID: {cluster}")
