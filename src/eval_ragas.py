import os
import pandas as pd

from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness

# ragas version compatibility:
try:
    from ragas.metrics import answer_relevance  # newer name (some versions)
except ImportError:
    from ragas.metrics import answer_relevancy as answer_relevance  # older/most common name


from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")

# --------------------------------------------------
# Load questions
# --------------------------------------------------
df = pd.read_csv("eval/questions.csv")

# --------------------------------------------------
# RAG pipeline
# --------------------------------------------------
client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=60.0)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
)

retriever = index.as_retriever(similarity_top_k=5)

questions = []
answers = []
contexts = []

for q in df["question"]:
    nodes = retriever.retrieve(q)

    ctx = [n.node.get_content() for n in nodes]
    answer = index.as_query_engine().query(q).response

    questions.append(q)
    answers.append(str(answer))
    contexts.append(ctx)

# --------------------------------------------------
# Build RAGAS dataset
# --------------------------------------------------
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
})

# --------------------------------------------------
# Evaluate
# --------------------------------------------------
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevance],
)

print(results)

# Save report
out_dir = "reports"
os.makedirs(out_dir, exist_ok=True)
results_df = results.to_pandas()
results_df.to_csv(f"{out_dir}/ragas_openai_report.csv", index=False)

print("\nSaved report to reports/ragas_openai_report.csv")
