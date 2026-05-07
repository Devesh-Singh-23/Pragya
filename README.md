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
- **💬 Per-Paper Chat History** – Each paper gets its own isolated, persistent conversation
- **🔍 Clickable Jargon Highlighting** – Technical terms are highlighted and clickable for instant Google search
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
                                                      ↓
                                          Chat Store (per-paper JSON)
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

## 💬 Per-Paper Chat History

Each paper maintains its own **isolated conversation** that is automatically saved and restored:

- **Automatic saving** – Chats are persisted to disk after every response
- **Seamless switching** – Switching papers in the sidebar instantly loads that paper's chat
- **Persistent across sessions** – Close and reopen the app; your conversations are still there
- **Clear chat** – Use the 🗑️ Clear Chat button in the sidebar to reset a paper's conversation
- **Storage** – Chats are stored as JSON files in `data/chat_history/`

## 🔍 Clickable Jargon Highlighting

Technical terms detected in assistant responses are **highlighted as clickable pills**:

- Clicking a highlighted term opens a **Google search** in a new tab
- Terms are detected from a curated jargon dictionary + LLM-identified definitions
- Each term is highlighted **only once** (first occurrence) to keep responses clean
- Jargon highlights are **persisted** with the chat history and restored on reload

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
│   ├── llm_client.py         # Ollama API client
│   ├── prompt_templates.py   # QA / Layman / Summary prompts
│   ├── layman_layer.py       # Jargon detection + readability scoring
│   ├── chat_store.py         # Per-paper chat persistence (JSON)
│   ├── rag_pipeline.py       # End-to-end orchestrator
│   └── utils.py              # Helpers
└── data/
    ├── uploads/              # Uploaded PDFs
    ├── chroma_db/            # Vector store persistence
    └── chat_history/         # Per-paper chat logs (JSON)
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm.model` | `llama3:latest` | Ollama model name |
| `llm.context_window` | `4096` | Context window size (lower = less RAM) |
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
