from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def build_similarity_matrix():
    df = pd.read_csv("data/online_retail.csv", encoding="latin1")

    # basic cleaning (minimum needed for recommender)
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

similarity_df = build_similarity_matrix()
