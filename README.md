# rag-quality-in-out  
## Quality-In, Quality-Out — RAG Optimizer

This repository implements a **Retrieval-Augmented Generation (RAG) pipeline**
focused on **grounded, traceable answers** over complex PDF documents.

The project follows the principle of **Quality-In, Quality-Out**:
accurate answers require structured ingestion, explicit grounding, and evaluation.

The system is designed as a **prototyping and evaluation sandbox**, using
open-source components and adapting to real-world constraints
(e.g. no paid APIs, limited resources).


## 🎯 Goals

- Structure-aware document ingestion (text + tables)
- Page-level, human-verifiable citations
- Transparent RAG outputs (retrieved extracts, not hallucinations)
- Automated evaluation design with RAGAS
- Fully runnable pipeline without paid embeddings


## 🧱 Architecture

PDF (WHO)
→ PyMuPDF ingestion (text + tables)
→ Qdrant vector store
→ LlamaIndex retriever
→ Streamlit UI (grounded extracts + citations)
→ RAGAS evaluation (design + integration)


## 📄 Ingestion & Grounding

- Text and tables are indexed separately (`content_type = text | table`)
- Tables are extracted with PyMuPDF and converted to Markdown
- Each chunk carries:
  - `source_document`
  - `page_number`
  - `content_type`

This ensures **explicit grounding and correct citation**.


## 🧠 Embeddings

To keep the pipeline fully runnable without external services, the project uses:

- **HashEmbedding** (custom, offline, deterministic)

This allows end-to-end execution and focuses the exercise on **RAG quality and architecture**, not model access.


## 🖥️ Streamlit UI

- Displays retrieved extracts (TEXT / TABLE)
- Orders results by source page
- Provides:
  - 📄 local PDF download
  - 🌐 official WHO web PDF link (page-level)


## 📊 Evaluation (RAGAS)

- Faithfulness
- Answer Relevance

The evaluation pipeline is fully implemented.
Metric execution requires an OpenAI API key with available quota.

How to Run

0. Set up the Python environment

From the project root, create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


The virtual environment is intentionally not committed.
Dependencies are fully reproducible via requirements.txt.

1. Start required services (Vector Store)

From the project root:

docker compose up -d


This starts Qdrant, used as the vector store.

Verify it is running:

docker ps

2. Prepare input documents

Create the input folder and download the sample WHO PDF:

mkdir -p data/samples
wget -O data/samples/worldhealthstatistics_2022.pdf \
https://cdn.who.int/media/docs/default-source/gho-documents/world-health-statistic-reports/worldhealthstatistics_2022.pdf

3. Run document ingestion (Quality-In)

Ingest the PDF with structure-aware parsing (text + tables):

python -m src.ingest


Expected output:

[ingest] worldhealthstatistics_2022.pdf: text_pages=..., tables=...
[ok] Indexed XXX documents into 'docs' at http://localhost:6333


At this point:

documents are parsed,

tables are preserved as structured content,

all chunks are indexed with page-level metadata.

4. Query the RAG pipeline via CLI (optional)

You can test retrieval and grounding directly from the terminal:

QUESTION="life expectancy" python -m src.rag_answer


This prints:

grounded text and table extracts,

page-level citations.

5. Launch the UI (Quality-Out)

Start the Streamlit application:

streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501


Open the URL shown in the terminal.

The UI allows you to:

ask questions,

inspect retrieved chunks (TEXT / TABLE),

verify sources via local PDF download or official WHO web links.

6. (Optional) Run automated evaluation with RAGAS

If an OpenAI API key with available quota is configured:

export OPENAI_API_KEY=your_key_here
python src/eval_ragas.py


This computes:

Faithfulness

Answer Relevance

Results are exported to the reports/ directory.

Note
At the time of submission, metric execution may be limited by API quota.
The evaluation pipeline is fully implemented and requires no code changes once quota is available.