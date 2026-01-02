import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()

# Local embeddings (same as ingestion/query scripts)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")

# Official WHO PDF URL (for web citations)
SOURCE_PDF_URL = os.getenv(
    "SOURCE_PDF_URL",
    "https://cdn.who.int/media/docs/default-source/gho-documents/"
    "world-health-statistic-reports/worldhealthstatistics_2022.pdf",
)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.set_page_config(page_title="WHO RAG Demo (Citations)", layout="wide")
st.title("WHO RAG Demo — Clickable Citations")

st.write(
    "This demo retrieves relevant chunks from the WHO (World Health Organization) PDF and shows grounded extracts "
    "with **page-level citations**. For each source you get:\n"
    "- a **download button** for the local PDF (Streamlit cannot open local files via plain links), and\n"
    "- a **clickable web link** to the official WHO PDF at the correct page.\n\n"
    "Tip: enable **Prefer tables** to visualize extracted tables (content_type=table)."
)

question = st.text_input("Question", value="life expectancy")
top_k = st.slider("Top-K shown results", min_value=1, max_value=10, value=5)
prefer_tables = st.checkbox("Prefer tables", value=False)

ask = st.button("Ask")

if ask:
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=60.0)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )

    # Retrieve more than needed, then filter/slice (helps pull in tables)
    retriever = index.as_retriever(similarity_top_k=top_k * 8)
    nodes = retriever.retrieve(question)

    # DEBUG: show content_type distribution in retrieved nodes (before filtering)
    types = [(n.node.metadata or {}).get("content_type", "MISSING") for n in nodes]
    st.caption(f"Retrieved types (raw): { {t: types.count(t) for t in set(types)} }")

    # Optional: prefer tables
    if prefer_tables:
        nodes = [n for n in nodes if (n.node.metadata or {}).get("content_type") == "table"]

    # Sort by page number (ascending) for readability
    nodes = sorted(nodes, key=lambda n: int((n.node.metadata or {}).get("page_number", 1_000_000_000)))

    # Keep only the requested amount after filtering/sorting
    nodes = nodes[:top_k]

    if not nodes:
        st.warning("No results. Try disabling 'Prefer tables' or changing the question.")
        st.stop()

    # --------------------------------------------------
    # Grounded answer
    # --------------------------------------------------
    st.subheader("Grounded answer (extracts)")
    for i, n in enumerate(nodes, start=1):
        meta = n.node.metadata or {}
        src = meta.get("source_document", "unknown")
        page = int(meta.get("page_number", -1))
        ctype = meta.get("content_type", "unknown")
        table_id = meta.get("table_id")

        content = (n.node.get_content() or "").strip()
        if len(content) > 1200:
            content = content[:1200] + "..."

        tag = f" [{ctype.upper()}]"
        extra = f" (table_id={table_id})" if (ctype == "table" and table_id is not None) else ""
        with st.expander(f"Chunk #{i}{tag} — {src}  p.{page}"):
            st.write(content)

    # --------------------------------------------------
    # Sources (local download + web link)
    # --------------------------------------------------
    st.subheader("Sources")
    seen = set()

    for n in nodes:
        meta = n.node.metadata or {}
        src = meta.get("source_document", "unknown")
        page = int(meta.get("page_number", -1))
        key = (src, page)

        if src == "unknown" or page == -1 or key in seen:
            continue
        seen.add(key)

        st.write(f"**{src} — p.{page}**")

        col1, col2 = st.columns(2)

        # Local PDF: download button (works reliably in Streamlit/Codespaces)
        with col1:
            local_path = Path("data/samples") / src
            if src and local_path.exists():
                with open(local_path, "rb") as f:
                    st.download_button(
                        label="📄 Local PDF (download)",
                        data=f,
                        file_name=src,
                        mime="application/pdf",
                        key=f"dl_{src}_{page}",
                    )
            else:
                st.write("Local PDF not available")

        # Web PDF: clickable link with page anchor
        with col2:
            web_link = f"{SOURCE_PDF_URL}#page={page}"
            st.markdown(f"[🌐 WHO Web PDF — open at page {page}]({web_link})")

        st.divider()
