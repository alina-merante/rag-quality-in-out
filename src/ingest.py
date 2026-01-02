import os
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
os.environ["HTTPX_USE_HTTP2"] = "0"

load_dotenv()

# Use local embeddings to avoid requiring OPENAI_API_KEY
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")


def pdf_to_page_documents(pdf_path: Path) -> list[Document]:
    doc = fitz.open(pdf_path)
    out: list[Document] = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text").strip()
        if not text:
            continue
        out.append(
            Document(
                text=text,
                metadata={
                    "source_document": pdf_path.name,
                    "page_number": i + 1,
                    "content_type": "text",
                },
            )
        )
    return out


def main():
    samples_dir = Path("data/samples")
    pdfs = sorted(samples_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("Nessun PDF in data/samples. Caricane uno e riprova.")

    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=60.0)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    all_docs: list[Document] = []
    for pdf in pdfs:
        docs = pdf_to_page_documents(pdf)
        print(f"[ingest] {pdf.name}: {len(docs)} pagine con testo")
        all_docs.extend(docs)

    if not all_docs:
        raise SystemExit("Non ho estratto testo: il PDF potrebbe essere una scansione (solo immagini).")

    VectorStoreIndex.from_documents(all_docs, storage_context=storage_context)
    print(f"[ok] Indicizzate {len(all_docs)} pagine nella collection '{COLLECTION_NAME}' su {QDRANT_URL}")


if __name__ == "__main__":
    main()
