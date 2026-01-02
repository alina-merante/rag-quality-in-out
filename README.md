# rag-quality-in-out
# Quality-In, Quality-Out — RAG Optimizer

This repository implements a **Retrieval-Augmented Generation (RAG) pipeline**
designed to ensure **grounded, traceable, and evaluable answers** over complex PDF documents.

The project focuses on the principle of **“Quality-In, Quality-Out”**:
high-quality answers are only possible if documents are ingested, structured,
retrieved, and evaluated correctly.

The system is built as a **prototyping and evaluation sandbox**, using open-source
components and low-code frameworks, with a strong emphasis on **grounding and metrics**.

## 🎯 Objectives

- Build a document ingestion pipeline that **understands structure**, not just text
- Ensure every answer is **explicitly grounded** in source documents
- Provide **page-level citations** that are human-verifiable
- Measure RAG quality with **automated metrics (RAGAS)** instead of assumptions
- Compare local/open-source approaches with **cloud-based LLM evaluation**


## 🧱 Architecture Overview

PDF Documents (WHO)
↓
Table-aware Ingestion (PyMuPDF)
↓
Vector Store (Qdrant)
↓
Retriever (LlamaIndex)
↓
Streamlit UI (Grounded Answers + Citations)
↓
RAGAS Evaluation (Faithfulness, Answer Relevance)


### Design Principles

- **Structure-aware ingestion**: text and tables are treated as different semantic units
- **Explicit grounding**: every retrieved chunk carries document and page metadata
- **Transparency over fluency**: answers are shown as grounded extracts
- **Evaluation-first mindset**: RAG quality is measured, not assumed


## 📄 Document Ingestion & Deep Understanding

### Text and Table Separation

PDFs are ingested using **PyMuPDF**.

For each page:
- Narrative text is indexed as `content_type = "text"`
- Tables are detected with `find_tables()`, converted to Markdown, and indexed as
  `content_type = "table"`

Each indexed node includes:
- `source_document`
- `page_number`
- `content_type` (`text` or `table`)

This enables:
- correct identification of tables as structured data,
- explicit retrieval of tabular information,
- precise citation of the source page.

This step addresses a common RAG failure mode, where tables are flattened into text
and lose semantic structure.


## 🔎 Retrieval Strategy

Retrieval is implemented with **LlamaIndex** on top of **Qdrant**.

Key aspects:
- High-recall retrieval (retrieving more candidates than shown)
- Optional **table-first retrieval** via UI toggle
- Post-retrieval sorting by **page number** for readability and traceability

This avoids relying solely on vector similarity, which often favors narrative text
over structured tables.


## 🖥️ Streamlit UI — Grounded Answers

The Streamlit interface is designed to **make grounding explicit**.

### Features

- Answers are shown as **retrieved extracts**, not hallucinated summaries
- Each chunk is labeled as `[TEXT]` or `[TABLE]`
- Results are ordered by source page
- A dedicated **Sources** section shows citations

### Citations

For every cited page:
- 📄 **Local PDF** is available via a download button  
  (Streamlit does not serve static files directly)
- 🌐 **Official web PDF** opens the authoritative source at the exact page

This allows instant human verification of the answer.


## 📊 Automated Evaluation with RAGAS

The project includes a complete **RAGAS evaluation pipeline**.

### Metrics

- **Faithfulness**  
  Measures whether the answer is supported by the retrieved contexts.
- **Answer Relevance**  
  Measures how well the answer addresses the question.

### Evaluation Pipeline

- Questions are defined in `eval/questions.csv`
- Contexts are retrieved using the same RAG pipeline as the UI
- An LLM is used as a judge to compute the metrics
- Results are exported as CSV to the `reports/` directory

Script:
```bash
python src/eval_ragas.py


Note on Execution

At the time of submission, the evaluation pipeline executes correctly up to the LLM
call, but metric computation could not be completed due to insufficient OpenAI API quota.

This limitation is external to the implementation.
Once quota is available, the pipeline produces metric scores without any code changes.

The evaluation design and integration are fully implemented and documented.

🚀 How to Run
1. Start the Vector Store
docker compose up -d

2. Ingest Documents
python src/ingest.py

3. Run the UI
streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501

4. (Optional) Run Evaluation
python src/eval_ragas.py

Requires a valid OPENAI_API_KEY.