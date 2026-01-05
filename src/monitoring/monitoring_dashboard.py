import streamlit as st
import json
import pandas as pd

st.title("📈 Recommender System Monitoring")

with open("metrics/metrics.jsonl") as f:
    records = [json.loads(line) for line in f]

df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"])

st.subheader("Embedding Drift Over Time")
st.line_chart(df.set_index("timestamp")["embedding_drift"])

st.subheader("Category PSI Over Time")
st.line_chart(df.set_index("timestamp")["category_psi"])
