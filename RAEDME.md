# 🛍️ Similar Product Recommendation System

**CLIP • FAISS • MMR • FastAPI • Streamlit**

An end-to-end **production-style similarity recommendation system** that retrieves visually and semantically similar products using deep embeddings, approximate nearest neighbor search, and diversity-aware re-ranking.

This project demonstrates **real-world ML system design**, not just model training.

---

## 🔍 Problem Statement

Given a product (image + metadata), recommend **similar products** that are:

* Visually and semantically relevant
* Fast to retrieve at scale
* Explainable to stakeholders
* Tunable for **exploration vs business control**

---

## 🚀 Key Features

* **CLIP image embeddings** for semantic similarity
* **Hybrid embeddings** (image + text + category signal)
* **FAISS** for fast approximate nearest neighbor (ANN) retrieval
* **MMR (Maximal Marginal Relevance)** for diversity control
* **Soft category similarity** (semantic) + **hard category match** (business)
* **FastAPI** backend for production-style serving
* **Streamlit UI** as a thin client (no ML logic in UI)
* Clear evaluation and explainability

---

## 🧠 System Architecture

```
                 ┌────────────────────┐
                 │   Product Dataset   │
                 │ (Images + Metadata) │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │  Data Ingestion &   │
                 │  Preprocessing      │
                 │ (validation, resize)│
                 └─────────┬──────────┘
                           │
          ┌────────────────▼─────────────────┐
          │     Embedding Generation          │
          │  - CLIP Image Embeddings          │
          │  - CLIP Text Embeddings           │
          │  - Hybrid Embeddings              │
          └────────────────┬─────────────────┘
                           │
                 ┌─────────▼──────────┐
                 │   FAISS Index       │
                 │ (ANN Recall Layer)  │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │   MMR Re-Ranking    │
                 │ (Relevance vs       │
                 │  Diversity Control) │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │  Semantic + Business│
                 │  Re-Ranking         │
                 │ (category signals)  │
                 └─────────┬──────────┘
                           │
              ┌────────────▼────────────┐
              │        FastAPI           │
              │   /recommend endpoint    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │       Streamlit UI       │
              │  (Visualization +        │
              │   Explainability)        │
              └─────────────────────────┘
```

---

## 🧩 Dataset

**Source:** Kaggle Fashion Product Images Dataset

**Contents:**

* ~44,000 fashion product images
* Metadata fields:

  * `product_id`
  * `articleType` (category)
  * `baseColour`
  * `season`
  * `usage`
  * `productDisplayName`

**Why this dataset?**

* Realistic noise (missing images, inconsistent labels)
* Requires ingestion validation
* Forces robust system design (not toy-clean data)

---

## 📊 Recommendation Models

### 1️⃣ CLIP (Exploratory)

* Uses CLIP image embeddings
* Optimized for **semantic similarity**
* Allows cross-category recommendations
* Higher diversity, lower business control

### 2️⃣ Hybrid (Business-Controlled)

* Combines:

  * Image similarity
  * Text/category semantics
* Higher category consistency
* Lower diversity (intentional)

---

## 🎯 Diversity Control (MMR)

We apply **Maximal Marginal Relevance (MMR)** after FAISS recall:

[
MMR(d) = \lambda \cdot sim(query, d) - (1 - \lambda) \cdot \max sim(d, selected)
]

* `λ ≈ 0.9` → highly relevant, less diverse
* `λ ≈ 0.5` → balanced
* `λ ≈ 0.1` → exploratory, diverse

Exposed directly in the UI.

---

## 🧠 Explainability

Each recommendation includes:

* **ANN similarity score**
* **Soft category similarity** (semantic closeness via CLIP text embeddings)
* **Hard category match** (exact label equality)

This avoids misleading metrics like “Precision@5 = 0.0” for valid semantic results.

---


## 🔬 Model Comparison & Evaluation

This project evaluates **multiple embedding strategies** to understand trade-offs between relevance, diversity, latency, and business control.

The goal is **not** to find a single “best” model, but to understand **when each model should be used**.

---

## 🤖 Evaluated Models

### 1️⃣ ResNet (Baseline)

**Description**

* CNN trained for image classification
* Used as a feature extractor (final pooling layer)

**Why included**

* Strong visual baseline
* Common in legacy vision systems
* Helps quantify gains from modern multimodal models

**Strengths**

* Good color and texture matching
* Stable and predictable behavior

**Limitations**

* No semantic understanding
* Fails on style, intent, or abstract similarity

---

### 2️⃣ EfficientNet

**Description**

* Parameter-efficient CNN with compound scaling
* Produces stronger visual embeddings than ResNet

**Why included**

* Better accuracy-latency trade-off
* Common upgrade path in production vision systems

**Strengths**

* Improved visual discrimination
* Lower latency than ResNet

**Limitations**

* Still purely visual
* Cannot reason about semantics or category intent

---

### 3️⃣ CLIP (Image-Only)

**Description**

* Vision–language model trained on image–text pairs
* Image embeddings capture **semantic meaning**

**Why included**

* Represents modern retrieval systems
* Enables cross-category and style-based similarity

**Strengths**

* Strong semantic similarity
* Handles style, intent, and abstract concepts
* Very fast ANN retrieval

**Limitations**

* Ignores business taxonomy
* Can recommend “related” but different categories

---

### 4️⃣ Hybrid (CLIP + Category Signal)

**Description**

* Combines CLIP embeddings with category semantics
* Adds business awareness without retraining CLIP

**Why included**

* Mirrors real-world recommender constraints
* Balances exploration and control

**Strengths**

* High relevance
* Strong category consistency
* Predictable for merchandising

**Limitations**

* Reduced diversity
* Less exploratory by design

---

## 📊 Evaluation Metrics

We evaluate models across **relevance, diversity, and system performance**.

### 🔹 Precision@5 (Category Proxy)

* Measures how many recommended items share the same category
* Useful as a **business proxy**, not absolute relevance

⚠️ Limitation:
Semantic models (CLIP) may score low despite valid recommendations.

---

### 🔹 Category Consistency

* Fraction of recommendations matching the query category
* Indicates catalog alignment and business control

---

### 🔹 ILD@5 (Intra-List Diversity)

* Measures average dissimilarity among recommended items
* Higher = more diverse results

---

### 🔹 Category Spread@5

* Number of unique categories in the recommendation list
* Captures exploration across catalog structure

---

### 🔹 Avg Query Latency (ms)

* End-to-end retrieval latency
* Important for real-time systems

---

## 📈 Final Evaluation Results

| Model        | Precision@5 | Category Consistency | ILD@5 | Category Spread@5 | Avg Latency (ms) |
| ------------ | ----------- | -------------------- | ----- | ----------------- | ---------------- |
| ResNet       | 0.764       | 0.820                | 0.068 | 0.331             | 31.22            |
| EfficientNet | 0.808       | 0.860                | 0.194 | 0.298             | 19.41            |
| CLIP         | 0.816       | 0.872                | 0.061 | 0.298             | 7.82             |
| **Hybrid**   | **0.961**   | **0.988**            | 0.069 | 0.227             | **7.80**         |

---

## 🧠 How to Interpret These Results (Important)

### Why CLIP can have lower Precision@5 but still be correct

* CLIP optimizes **semantic similarity**, not taxonomy
* Recommending “Sunglasses” for “Watches” can be valid
* Hard category metrics underestimate semantic relevance

➡️ This is why we expose **soft category similarity**.

---

### Why Hybrid scores highest on Precision@5

* Category signal is explicitly injected
* This is intentional and business-driven
* Not “better ML” — **better alignment with constraints**

---

### Why ILD is lower for Hybrid

* Business control reduces exploration
* This is a **trade-off**, not a failure

---

### Why MMR is critical

* FAISS recall returns near-duplicates
* MMR explicitly balances:

  * relevance
  * redundancy
  * diversity

This allows **runtime tuning** without retraining.

---

## 🎯 Model Selection Guidelines

| Use Case                            | Recommended Model |
| ----------------------------------- | ----------------- |
| Visual similarity only              | EfficientNet      |
| Style / semantic discovery          | CLIP              |
| Business-controlled recommendations | Hybrid            |
| Exploration vs control tuning       | Any + MMR         |

---

## 🧠 Key Insight

There is **no universally best model**.

A good recommender system:

* exposes trade-offs
* explains behavior
* adapts to business goals

---

## 🏗️ Project Structure
```
similar-product-recommendation/
│
├── data/
│   ├── raw/                     # Original dataset (images + CSV)
│   └── processed/               # Validated images + cleaned metadata
│
├── artifacts/                   # Generated model artifacts
│   ├── resnet/                  # ResNet embeddings + product IDs
│   ├── efficientnet/            # EfficientNet embeddings + product IDs
│   ├── clip_image/              # CLIP image embeddings
│   ├── clip_text/               # CLIP text embeddings
│   └── hybrid/                  # Hybrid embeddings (image + category)
│
├── src/
│   ├── ingestion/               # Data ingestion & validation
│   │   ├── kaggle_ingest.py
│   │   ├── validate.py
│   │   └── normalize.py
│   │
│   ├── modeling/                # Embedding generation
│   │   ├── resnet/
│   │   ├── efficientnet/
│   │   ├── clip_image/
│   │   ├── clip_text/
│   │   └── hybrid/
│   │
│   ├── evaluation/              # Offline evaluation & metrics
│   │   ├── metrics.py
│   │   ├── diversity.py
│   │   └── evaluate_models.py
│   │
│   ├── indexing/                # Index construction (offline)
│   │   └── build_faiss_index.py
│   │
│   └── api/                     # Online serving layer
│       ├── main.py              # FastAPI application
│       ├── faiss_index.py       # FAISS wrapper + ID mapping
│       ├── rerank.py            # Semantic + business re-ranking
│       ├── mmr.py               # Diversity-aware re-ranking (MMR)
│       └── schemas.py           # API request/response models
│
├── streamlit_app.py             # Streamlit UI (thin client)
├── requirements.txt             # Python dependencies
├── noteook                      # Experimental notebook
├── tests/                       # Optional: unit / integration tests
│   └── test_api.py
│
└── README.md                    # Project documentation
```

---

## ▶️ How to Run (End-to-End)

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Generate embeddings (one-time)

```bash
python src/ingestion/kaggle_ingest.py
python src/modeling/clip_image/batch_embed.py
python src/modeling/clip_text/batch_embed.py
python src/modeling/hybrid/batch_embed.py
```

### 3️⃣ Start FastAPI backend

```bash
uvicorn src.api.main:app --reload
```

Verify:

```
http://127.0.0.1:8000/docs
```

### 4️⃣ Start Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open:

```
http://localhost:8501
```

---

## 🧪 API Example

```http
GET /recommend?product_id=15970&model=clip&top_k=5&lambda_diversity=0.7
```

Returns:

```json
{
  "query_product_id": "15970",
  "model": "clip",
  "top_k": 5,
  "results": [
    {
      "product_id": "39386",
      "score": 0.812,
      "ann_score": 0.94,
      "category_similarity": 0.67,
      "category_match": false
    }
  ]
}
```

---

## 🎤 Summary

> “Designed a two-stage recommender system: FAISS for fast recall, followed by MMR and semantic re-ranking for controllable diversity and explainability. The UI is a thin client consuming a FastAPI service, mirroring production ML systems.”

---

## 🚧 Future Improvements

* Online A/B testing
* User interaction feedback loop
* Learned re-ranking model
* Caching layer (Redis)
* Cloud deployment (ECS / GKE)

---

## ✅ Key Takeaway

This project is not about “training a model”.

It demonstrates:

* **System design**
* **Trade-off reasoning**
* **Production ML thinking**
* **Explainability over blind metrics**

