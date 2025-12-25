import streamlit as st
import pandas as pd
import requests

# =======================
# CONFIG
# =======================
API_URL = "http://127.0.0.1:8000/recommend"
METADATA_PATH = "data/processed/metadata.csv"
K = 5

# =======================
# LOAD METADATA
# =======================
@st.cache_data
def load_metadata():
    df = pd.read_csv(METADATA_PATH, dtype={"product_id": str})
    return df.set_index("product_id")

df = load_metadata()

# =======================
# UI HEADER
# =======================
st.title("🛍️ Similar Product Recommendation System")
# st.caption(
#     "Streamlit UI → FastAPI → FAISS → MMR → Explainability (Hard + Soft Category Signals)"
# )

# =======================
# USER INPUTS
# =======================
product_id = st.selectbox("Select Product ID", df.index.tolist())

model_label = st.radio(
    "Recommendation Mode",
    ["CLIP (Exploratory)", "Hybrid (Business-Controlled)"]
)

model = "clip" if model_label.startswith("CLIP") else "hybrid"

lambda_div = st.slider(
    "MMR λ (Relevance ↔ Diversity)",
    min_value=0.1,
    max_value=0.9,
    value=0.7,
    step=0.1
)

# =======================
# API CALL (SAFE)
# =======================
params = {
    "product_id": product_id,
    "model": model,
    "top_k": K,
    "lambda_diversity": lambda_div
}
st.caption(
    "MMR (Maximal Marginal Relevance) controls redundancy. "
    "Lower λ increases diversity, higher λ favors closer matches."
)
try:
    response = requests.get(API_URL, params=params, timeout=5)
except requests.exceptions.ConnectionError:
    st.error("❌ FastAPI backend not running on port 8000")
    st.stop()

if response.status_code != 200:
    st.error(f"API Error ({response.status_code})")
    st.text(response.text)
    st.stop()

try:
    data = response.json()
except ValueError:
    st.error("❌ API returned non-JSON response")
    st.text(response.text)
    st.stop()

results = data["results"]

# =======================
# QUERY PRODUCT
# =======================
st.subheader("Query Product")
st.image(df.loc[product_id]["image_path"], width=200)
query_category = df.loc[product_id]["category"]
st.write("**Category:**", query_category)

# =======================
# RESULTS
# =======================
st.subheader("Recommended Products")

cols = st.columns(K)
for col, item in zip(cols, results):
    pid = item["product_id"]

    with col:
        st.image(df.loc[pid]["image_path"], width=200)
        st.caption(f"Score: {item['score']:.3f}")
        st.caption(f"Category: {df.loc[pid]['category']}")
        st.caption(f"Category similarity: {item['category_similarity']:.2f}")

        if item["category_match"]:
            st.caption("✅ Same category")
        else:
            st.caption("🔍 Related category")

# =======================
# EXPLAINABILITY METRICS
# =======================
st.subheader("Why these results?")

exact_matches = sum(r["category_match"] for r in results)
avg_cat_sim = sum(r["category_similarity"] for r in results) / K

st.metric("Exact Category Match@5", round(exact_matches / K, 3))
st.metric("Avg Category Similarity@5", round(avg_cat_sim, 3))

# =======================
# EXPLANATION TEXT
# =======================
if model == "clip":
    st.info(
        "CLIP focuses on semantic similarity and allows cross-category exploration. "
        "Low exact matches with moderate category similarity are expected."
    )
else:
    st.warning(
        "Hybrid enforces business constraints. "
        "Exact category matches are prioritized, reducing exploration."
    )



# =======================
# FOOTER
# =======================
st.caption(
    "The UI is a thin client. All retrieval, ranking, and explainability "
    "are handled by the FastAPI backend."
)
