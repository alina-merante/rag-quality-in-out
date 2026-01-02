import os
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv

# Workaround for some environments (Codespaces) where httpx/http2 can be flaky
os.environ["HTTPX_USE_HTTP2"] = "0"

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Local embeddings (no OpenAI key needed)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")

# Optional: official URL for web citations
SOURCE_PDF_URL = os.getenv(
    "SOURCE_PDF_URL",
    "https://cdn.who.int/media/docs/default-source/gho-documents/"
    "world-health-statistic-reports/worldhealthstatistics_2022.pdf",
)


def table_to_markdown(table) -> str:
    """
    Convert a PyMuPDF table to Markdown.
    We keep it simple + readable for LLM/RAG.
    """
    rows = table.extract()
    if not rows:
        return ""

    # Normalize row lengths
    max_cols = max(len(r) for r in rows)
    norm = []
    for r in rows:
        r = list(r) + [""] * (max_cols - len(r))
        norm.append([("" if c is None else str(c)).strip() for c in r])

    header = norm[0]
    body = norm[1:] if len(norm) > 1 else []

    # Markdown table
    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * max_cols) + " |")
    for r in body:
        md.append("| " + " | ".join(r) + " |")

    return "\n".join(md)


def pdf_to_documents(pdf_path: Path) -> list[Document]:
    """
    Create Documents for:
    - text per page (content_type=text)
    - each table per page (content_type=table, with table_id)
    """
    doc = fitz.open(pdf_path)
    out: list[Document] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        # ---- TEXT ----
        text = page.get_text("text").strip()
        if text:
            out.append(
                Document(
                    text=text,
                    metadata={
                        "source_document": pdf_path.name,
                        "page_number": page_number,
                        "content_type": "text",
                        "source_url": SOURCE_PDF_URL,
                    },
                )
            )

        # ---- TABLES ----
        try:
            tables = page.find_tables()
            if tables and tables.tables:
                for t_i, t in enumerate(tables.tables, start=1):
                    md = table_to_markdown(t).strip()
                    if not md:
                        continue
                    # Add a small prefix so retrieval knows it's a table
                    table_text = f"TABLE (page {page_number}, table {t_i})\n{md}"
                    out.append(
                        Document(
                            text=table_text,
                            metadata={
                                "source_document": pdf_path.name,
                                "page_number": page_number,
                                "content_type": "table",
                                "table_id": t_i,
                                "source_url": SOURCE_PDF_URL,
                            },
                        )
                    )
        except Exception:
            # If table extraction fails on some pages, we still keep text
            pass

    return out


def main():
    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        raise SystemExit("Missing folder data/samples. Put your PDF(s) there.")

    pdfs = sorted(samples_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in data/samples. Add at least one PDF.")

    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=60.0)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    all_docs: list[Document] = []
    for pdf in pdfs:
        docs = pdf_to_documents(pdf)
        # Count text vs tables for logging
        text_count = sum(1 for d in docs if d.metadata.get("content_type") == "text")
        table_count = sum(1 for d in docs if d.metadata.get("content_type") == "table")
        print(f"[ingest] {pdf.name}: text_pages={text_count}, tables={table_count}")
        all_docs.extend(docs)

    if not all_docs:
        raise SystemExit("No content extracted from PDFs (are they scanned images?).")

    VectorStoreIndex.from_documents(all_docs, storage_context=storage_context)
    print(f"[ok] Indexed {len(all_docs)} documents into '{COLLECTION_NAME}' at {QDRANT_URL}")


if __name__ == "__main__":
    main()
