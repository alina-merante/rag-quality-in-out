import os
from dotenv import load_dotenv

from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.hash_embedding import HashEmbedding

load_dotenv()

# Offline embedding (must match ingestion)
Settings.embed_model = HashEmbedding(dim=384)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")


def make_clickable_source(source_document: str, page_number: int) -> str:
    # Link to local PDF path with page anchor (works in many viewers)
    return f"[{source_document} – p.{page_number}](data/samples/{source_document}#page={page_number})"


def main():
    question = os.getenv("QUESTION") or "What does the report say about life expectancy?"

    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=60.0)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(question)

    if not nodes:
        print("No results.")
        return

    # Simple grounded "extractive" answer: show the most relevant snippets
    print("\n=== QUESTION ===")
    print(question)

    print("\n=== GROUNDED ANSWER (extracts) ===")
    for i, n in enumerate(nodes, start=1):
        meta = n.node.metadata or {}
        text = (n.node.get_content() or "").strip().replace("\n", " ")
        if len(text) > 350:
            text = text[:350] + "..."
        print(f"{i}. {text}")

    # Clickable citations (deduplicated)
    seen = set()
    sources = []
    for n in nodes:
        meta = n.node.metadata or {}
        src = meta.get("source_document", "unknown")
        page = int(meta.get("page_number", -1))
        key = (src, page)
        if src != "unknown" and page != -1 and key not in seen:
            seen.add(key)
            sources.append(make_clickable_source(src, page))

    print("\n=== SOURCES (clickable) ===")
    for s in sources:
        print("-", s)


if __name__ == "__main__":
    main()
