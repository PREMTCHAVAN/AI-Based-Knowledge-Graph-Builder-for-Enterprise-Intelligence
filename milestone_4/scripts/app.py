# app.py
# Run with: streamlit run app.py

import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Graph Explorer",
    page_icon="🔍",
    layout="wide"
)

st.title("AI Knowledge Graph Explorer")
st.caption("Semantic + keyword hybrid search over your enterprise graph")

# ─────────────────────────────────────────────
# LOAD EVERYTHING ONCE (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    with open(r"C:\Users\91985\OneDrive\Documents\Infosys Internship\AI-Based-Knowledge-Graph-Builder-for-Enterprise-Intelligence\milestone_4\data\results.json", "r") as f:
        all_results = json.load(f)

    texts = [r["document"] for r in all_results if len(r["entities"]) > 0]
    valid_results = [r for r in all_results if len(r["entities"]) > 0]

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embed_model.encode(texts, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    tokenized = [doc.lower().split() for doc in texts]
    bm25 = BM25Okapi(tokenized)

    return texts, valid_results, embed_model, index, bm25

texts, valid_results, embed_model, faiss_index, bm25 = load_pipeline()

# ─────────────────────────────────────────────
# SEARCH FUNCTIONS
# ─────────────────────────────────────────────
def semantic_search(query, top_k=10):
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_vec, top_k)
    return list(indices[0]), list(distances[0])

def keyword_search(query, top_k=10):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:top_k]
    return list(ranked), [float(scores[i]) for i in ranked]

def hybrid_search(query, top_k=5):
    sem_idx, sem_scores = semantic_search(query, top_k=top_k * 2)
    kw_idx, kw_scores   = keyword_search(query,  top_k=top_k * 2)

    seen = set()
    combined = []

    for idx, score in zip(sem_idx, sem_scores):
        if idx not in seen:
            seen.add(idx)
            combined.append({
                "text":          texts[idx],
                "entities":      valid_results[idx]["entities"],
                "relationships": valid_results[idx]["relationships"],
                "sem_score":     round(float(score), 4),
                "kw_score":      0.0,
                "method":        "semantic"
            })

    for idx, score in zip(kw_idx, kw_scores):
        if idx not in seen:
            seen.add(idx)
            combined.append({
                "text":          texts[idx],
                "entities":      valid_results[idx]["entities"],
                "relationships": valid_results[idx]["relationships"],
                "sem_score":     0.0,
                "kw_score":      round(float(score), 4),
                "method":        "keyword"
            })

    return combined[:top_k]

# ─────────────────────────────────────────────
# RAG FUNCTION
# ─────────────────────────────────────────────
def generate_rag_answer(query, context_docs, api_key):
    client  = Groq(api_key=api_key)
    context = "\n".join([f"- {d['text']}" for d in context_docs])
    prompt  = f"""You are a sales intelligence assistant.
Answer the question using ONLY the context below. Be concise.
If the answer is not in the context, say: "Not enough data to answer."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    top_k = st.slider("Number of results", min_value=3, max_value=15, value=5)

    st.divider()

    use_rag  = st.toggle("Generate AI answer (RAG)", value=False)
    groq_key = ""
    if use_rag:
        groq_key = st.text_input("Groq API key", type="password",
                                  placeholder="gsk_...")
        if not groq_key:
            st.warning("Enter your Groq API key to enable RAG.")

    st.divider()

    show_entities  = st.toggle("Show extracted entities",  value=True)
    show_relations = st.toggle("Show relationships",       value=False)

    st.divider()
    st.metric("Documents indexed", len(texts))

# ─────────────────────────────────────────────
# SEARCH BAR
# ─────────────────────────────────────────────
query = st.text_input(
    "Enter your question",
    placeholder="e.g. Which customers bought Beauty products from the North region?",
)

col1, col2 = st.columns([1, 5])
with col1:
    search_btn = st.button("Search", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if search_btn and query.strip():

    with st.spinner("Searching..."):
        results = hybrid_search(query, top_k=top_k)

    st.markdown(f"**{len(results)} results** for: *{query}*")
    st.divider()

    # RAG answer
    if use_rag and groq_key:
        with st.spinner("Generating AI answer..."):
            answer = generate_rag_answer(query, results, groq_key)
        st.subheader("AI Answer")
        st.success(answer)
        st.divider()

    # Result cards
    for i, doc in enumerate(results):
        with st.expander(f"Result {i+1} — {doc['method'].upper()}", expanded=(i < 3)):
            st.write(doc["text"])

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Semantic score", round(doc["sem_score"], 3))
            with col_b:
                st.metric("Keyword score",  round(doc["kw_score"],  3))
            with col_c:
                st.metric("Entities found", len(doc["entities"]))

            if show_entities and doc["entities"]:
                st.markdown("**Entities**")
                color_map = {
                    "Customer": "🟣", "Product": "🟢",
                    "Region":   "🔵", "Category": "🟡",
                }
                entity_cols = st.columns(min(len(doc["entities"]), 4))
                for j, ent in enumerate(doc["entities"]):
                    with entity_cols[j % 4]:
                        st.caption(f"{color_map.get(ent['type'], '⚪')} {ent['type']}")
                        st.write(ent["name"])

            if show_relations and doc["relationships"]:
                st.markdown("**Relationships**")
                for rel in doc["relationships"]:
                    st.code(
                        f"{rel['source']}  →[{rel['relation']}]→  {rel['target']}",
                        language=None
                    )

elif search_btn and not query.strip():
    st.warning("Please enter a search query.")

# ─────────────────────────────────────────────
# EXAMPLE QUERIES
# ─────────────────────────────────────────────
st.divider()
st.caption("Try these example queries:")
examples = [
    "customers who bought beauty products",
    "top sales in North region",
    "products in Electronics category",
    "orders with high sales amount",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    with col:
        st.button(ex, use_container_width=True)