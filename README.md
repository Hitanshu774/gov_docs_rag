# 🏛️ Government Documents RAG System

A Retrieval-Augmented Generation (RAG) pipeline for querying and reasoning over US federal government documents — including GAO reports, DoD policy documents, Inspector General reports, and congressional research materials.

Built with LangChain, ChromaDB, HuggingFace Embeddings, and OpenRouter-hosted LLMs. Includes a full custom evaluation framework measuring retrieval quality (MRR, NDCG) and answer quality (accuracy, completeness, relevance).

---

## ✨ Features

- **End-to-end RAG Pipeline** — JSONL corpus ingestion → semantic chunking → embedding → ChromaDB vector store → similarity retrieval → LLM answer generation
- **Domain-specific QA** — handles complex policy, regulatory, and investigative queries over government documents
- **Custom Evaluation Framework** — `RAGEvaluator` with keyword scoring, semantic overlap, length ratio, MRR, NDCG, and LLM-as-judge scoring
- **Multi-model Support** — swap LLMs via OpenRouter (Mistral Devstral, Mistral 7B Instruct, and others) without changing pipeline code
- **Gradio Chat Interface** — interactive web UI for querying the document corpus in real-time
- **Detailed Evaluation Reports** — per-question and per-category breakdown saved as structured JSON

---

## 🗂️ Project Structure

```
gov_docs_rag/
├── one.ipynb                        # Main RAG pipeline notebook
├── rag_evaluator.py                 # Custom RAG evaluation framework
├── corpus.jsonl                     # Government document corpus (~49MB)
├── tests.jsonl                      # Evaluation test set (questions, keywords, reference answers)
├── detailed_evaluation_results.json # Evaluation output with full metrics
├── vector_db/                       # Persisted ChromaDB vector store
├── .env                             # API keys (not committed)
└── .gitignore
```

---

## 🏗️ Architecture

```
corpus.jsonl
     │
     ▼
JSONLoader (LangChain)
     │  Loads each line as a Document
     ▼
RecursiveCharacterTextSplitter
     │  chunk_size=500, overlap=50
     ▼
HuggingFace Embeddings
     │  BAAI/bge-large-en-v1.5
     ▼
ChromaDB Vector Store (persisted)
     │
     ├──────────────────────────────────────┐
     │                                      │
[User Query]                          [RAGEvaluator]
     │                                      │
Similarity Retriever (k=5)           Retrieval Metrics
     │                                  MRR, NDCG
     ▼                                  Keyword Coverage
Context Assembly
     │
System Prompt + Context + Query
     │
     ▼
OpenRouter LLM (Mistral via OpenRouter API)
     │
     ▼
Generated Answer → Gradio UI / Evaluation
```

---

## 📊 Evaluation Results

Evaluated on 10 complex government policy questions across categories including `direct_fact` and multi-hop reasoning queries.

| Metric | Score |
|---|---|
| **Mean Reciprocal Rank (MRR)** | **0.90** |
| **Average NDCG** | 2.05 |
| **Avg Keyword Coverage** | 48.7% |
| **Avg Completeness (0–5)** | 3.0 |
| **Avg Accuracy (0–5)** | 1.62 |
| **Avg Relevance (0–5)** | 0.50 |

> MRR of 0.90 indicates the correct document is retrieved as the top result in 90% of queries — strong retrieval performance. Answer quality metrics reflect the challenge of open-domain government policy QA and scope for LLM prompt tuning.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Document Loading | LangChain `JSONLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `BAAI/bge-large-en-v1.5` |
| Vector Store | ChromaDB (persisted) |
| LLM | Mistral Devstral / Mistral 7B via OpenRouter |
| Retrieval | LangChain similarity retriever (k=5) |
| UI | Gradio `ChatInterface` |
| Evaluation | Custom `RAGEvaluator` + LLM-as-judge |
| Language | Python 3.9+ |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Hitanshu774/gov_docs_rag.git
cd gov_docs_rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install langchain langchain-community langchain-chroma langchain-huggingface \
            langchain-openai langchain-google-genai chromadb gradio \
            python-dotenv sentence-transformers
```

### 4. Configure Environment Variables

```bash
cp .env .env.local
```

Edit `.env` and add your OpenRouter API key:

```env
api-key=your_openrouter_api_key_here
```

Get a free API key at [openrouter.ai](https://openrouter.ai) — free-tier models (Mistral 7B Instruct) are sufficient to run the pipeline.

### 5. Build the Vector Store (first run only)

In `one.ipynb`, uncomment the embedding block:

```python
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)
```

> ⚠️ First run downloads the embedding model (~1.3GB) and embeds the full corpus. This takes 10–20 minutes. Subsequent runs load directly from `vector_db/`.

### 6. Run the Notebook

```bash
jupyter notebook one.ipynb
```

Launch the Gradio chat UI by running the final cell:

```python
gr.ChatInterface(answer_question).launch()
```

---

## 🧪 Running Evaluations

The `RAGEvaluator` class in `rag_evaluator.py` supports plug-and-play evaluation against any answer function.

```python
from rag_evaluator import RAGEvaluator

evaluator = RAGEvaluator(tests_path="tests.jsonl")

report = evaluator.run_evaluation(
    answer_fn=answer_question,   # your RAG answer function
    verbose=True,
    keyword_pass_threshold=0.8
)

RAGEvaluator.print_report(report)
RAGEvaluator.save_results(report, "my_eval_results.json")
```

### Evaluation Metrics

| Metric | Description |
|---|---|
| **Keyword Score** | Fraction of expected keywords present in the generated answer |
| **Semantic Overlap** | Jaccard similarity between generated and reference answer token sets |
| **Length Ratio** | Penalizes answers that are too short or too long relative to the reference |
| **Combined Score** | `0.4 × keyword + 0.4 × semantic + 0.2 × length` |
| **MRR** | Mean Reciprocal Rank of first relevant document in retrieval results |
| **NDCG** | Normalized Discounted Cumulative Gain over retrieved document ranking |
| **LLM-as-Judge** | GPT/Mistral evaluates accuracy, completeness, and relevance on a 0–5 scale |

---

## 📄 Dataset

**Corpus (`corpus.jsonl`)** — ~49MB JSONL file where each line contains the `text` field of a US government document. Covers domains including:

- Defense Acquisition & DoD Housing (MHPI)
- Inspector General reports (ATF, USMS, DOJ OIG)
- Congressional Research Service documents
- Environmental Justice policy (Executive Order 12898)
- Senate nomination and confirmation procedures
- Afghanistan reconstruction oversight (SIGAR)

**Test Set (`tests.jsonl`)** — 10 evaluation questions with reference answers, keyword lists, and category labels. Questions are complex, multi-hop policy reasoning queries designed to stress-test retrieval and generation quality.

---

## 🔧 Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 500 | Token size per chunk |
| `chunk_overlap` | 50 | Overlap between consecutive chunks |
| `k` (retriever) | 5 | Number of documents retrieved per query |
| `temperature` | 0.3 | LLM sampling temperature |
| `max_tokens` | 512 | Max tokens in generated answer |
| Embedding model | `BAAI/bge-large-en-v1.5` | HuggingFace dense retrieval model |
| LLM | `mistralai/devstral-2512:free` | Via OpenRouter |

---

## 🔮 Future Enhancements

- [ ] Hybrid retrieval (dense + BM25 keyword search)
- [ ] Re-ranking layer (cross-encoder) for improved retrieval precision
- [ ] LangSmith tracing and prompt versioning for systematic experimentation
- [ ] Support for PDF ingestion alongside JSONL corpus
- [ ] Streamlit or FastAPI deployment for production serving
- [ ] Expand test set to 100+ questions with automated evaluation pipeline

---

## 👤 Author

**Hitanshu** — M.Tech Artificial Intelligence, IIIT Lucknow

Specializing in agentic AI systems, RAG pipelines, LangGraph, LangSmith, and LLM deployment.

[![GitHub](https://img.shields.io/badge/GitHub-Hitanshu774-black?logo=github)](https://github.com/Hitanshu774)

---

## ⚠️ Disclaimer

This system is intended for research and information retrieval purposes only. Government documents may be subject to usage restrictions. Always verify critical policy information with official government sources.
