import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Streamlit runs this file with `src/` as the script folder, so import local module directly.
from hash_embedding import HashEmbedding


# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()

# Offline embedding (must match ingestion + rag_answer)
Settings.embed_model = HashEmbedding(dim=384)

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
st.set_page_config(page_title="RAG Demo (Citations)", layout="wide")
st.title("RAG Demo — Grounded Extracts + Citations")

st.write(
    "This demo retrieves relevant chunks from the ingested PDF(s) in Qdrant and shows grounded extracts "
    "with **page-level citations**.\n\n"
    "For each source you get:\n"
    "- a **local PDF download** button, and\n"
    "- a **web link** to the official WHO PDF at the correct page.\n\n"
    "Tip: enable **Prefer tables** to prioritize extracted tables (content_type=table) without hiding text results."
)

question = st.text_input("Question", value="life expectancy")
top_k = st.slider("Top-K shown results", min_value=1, max_value=10, value=5)
prefer_tables = st.checkbox("Prefer tables", value=False)

ask = st.button("Ask")

if ask:
    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=60.0)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index wrapper around existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    # Retrieve more than needed, then re-rank (helps pull in tables)
    raw_k = top_k * (40 if prefer_tables else 12)
    retriever = index.as_retriever(similarity_top_k=raw_k)
    nodes = retriever.retrieve(question)

    # Split by type
    tables = [n for n in nodes if (n.node.metadata or {}).get("content_type") == "table"]
    texts = [n for n in nodes if (n.node.metadata or {}).get("content_type") != "table"]

    # Prefer tables = re-rank tables first, but NEVER filter out text completely
    if prefer_tables:
        nodes = tables + texts
    else:
        nodes = texts + tables  # keep text first by default

    # Sort for readability (tables first if prefer_tables, then page number)
    nodes = sorted(
        nodes,
        key=lambda n: (
            0 if (prefer_tables and (n.node.metadata or {}).get("content_type") == "table") else 1,
            int((n.node.metadata or {}).get("page_number", 1_000_000_000)),
        ),
    )

    # Keep only top_k after re-ranking/sorting
    nodes = nodes[:top_k]

    # Debug counters (useful to understand what retrieval returned)
    st.caption(f"Retrieved (raw): total={len(texts)+len(tables)} | tables={len(tables)} | text={len(texts)}")

    if not nodes:
        st.error("No results at all. Check that ingestion ran and Qdrant collection contains points.")
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
        with st.expander(f"Chunk #{i}{tag} — {src} p.{page}{extra}"):
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

        # Local PDF download
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

        # Web PDF link at page
        with col2:
            web_link = f"{SOURCE_PDF_URL}#page={page}"
            st.markdown(f"[🌐 WHO Web PDF — open at page {page}]({web_link})")

        st.divider()
