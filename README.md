# 🧠 Pragya – Research Papers, Simplified

A fully local RAG (Retrieval-Augmented Generation) system that helps you understand complex research papers using simple, everyday language.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Ollama](https://img.shields.io/badge/LLM-Ollama-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **🌟 Layman Explanations** – Get research explained like you're talking to a friend
- **🔬 Research QA** – Ask technical questions, get precise paper-backed answers
- **📋 Smart Summaries** – Structured overviews with key findings and implications
- **📄 Section-Aware Parsing** – Understands Abstract, Methodology, Results, etc.
- **🔒 100% Local** – No data leaves your machine. Ever.
- **⚡ Streaming Responses** – See answers appear in real-time

## 🏗️ Architecture

```
PDF → PyMuPDF Parser → Section-Aware Chunker → Embeddings (MiniLM)
                                                      ↓
User Question → Query Embedding → ChromaDB Search → Prompt Router
                                                      ↓
                                          Ollama (Llama3/Mistral)
                                                      ↓
                                          Layman Layer → Streamlit UI
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- 16GB RAM, 6GB+ VRAM recommended

### 1. Install Dependencies
```bash
cd Pragya
pip install -r requirements.txt
```

### 2. Pull an LLM Model
```bash
ollama pull llama3:8b-instruct-q4_K_M
```

### 3. Run Pragya
```bash
streamlit run app.py
```

### 4. Upload & Ask
1. Upload a PDF research paper from the sidebar
2. Select your mode (Layman / QA / Summary)
3. Ask questions in the chat!

## 📁 Project Structure

```
Pragya/
├── app.py                    # Streamlit UI entry point
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── pragya/
│   ├── pdf_parser.py         # Section-aware PDF extraction
│   ├── chunker.py            # Smart recursive chunking
│   ├── embeddings.py         # Sentence Transformers pipeline
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── retriever.py          # Similarity search
│   ├── llm_client.py         # Ollama API client
│   ├── prompt_templates.py   # QA / Layman / Summary prompts
│   ├── layman_layer.py       # Jargon detection + readability
│   ├── rag_pipeline.py       # End-to-end orchestrator
│   └── utils.py              # Helpers
└── data/
    ├── uploads/              # Uploaded PDFs
    └── chroma_db/            # Vector store persistence
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm.model` | `llama3:8b-instruct-q4_K_M` | Ollama model name |
| `chunking.chunk_size` | `512` | Characters per chunk |
| `retrieval.top_k` | `5` | Number of chunks retrieved |
| `readability.min_flesch_score` | `60` | Minimum readability for layman mode |

## 🧠 Response Modes

| Mode | Use Case | Style |
|------|----------|-------|
| 🌟 **Layman** | Understanding the paper | Simple analogies, no jargon |
| 🔬 **Research QA** | Specific technical questions | Precise, paper-level detail |
| 📋 **Summary** | Quick overview | Structured with key findings |

## 📊 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| VRAM | 4 GB | 6+ GB |
| Storage | 5 GB | 10 GB |
| CPU | 4 cores | 8 cores |

## 📜 License

MIT License – Free for academic and personal use.
