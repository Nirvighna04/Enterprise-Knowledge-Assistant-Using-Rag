# 🧠 RAG Knowledge Assistant

> Answer questions **strictly from your own documents** — powered by local LLMs via Ollama. Zero hallucinations, zero API costs.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 What is this?

A **Retrieval-Augmented Generation (RAG)** chatbot that:
- Lets users **upload PDF, DOCX, and TXT** documents
- Answers questions **only from those documents**
- Returns **"Not found"** if the answer isn't in your docs
- Shows **citations** (document name, page, line number)
- Runs **100% locally** — no OpenAI, no API keys

---

## 🖼️ App Preview

> Login → Upload Documents → Ask Questions → Get Cited Answers

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend + Backend | Streamlit |
| LLM | Ollama (llama3 / mistral / any local model) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS (local persistent) |
| Database | SQLite + SQLAlchemy |
| Document Parsing | pdfplumber, python-docx |

---

## 📂 Project Structure
```
rag-knowledge-assistant/
├── app.py            # Streamlit UI — Login, Chatbot, History pages
├── auth.py           # Register & Login logic
├── rag.py            # Ingestion, chunking, FAISS, Ollama call
├── db.py             # SQLAlchemy models + DB helpers
├── vector_store/     # FAISS index (auto-created on first upload)
├── data/             # Uploaded documents (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/rag-knowledge-assistant.git
cd rag-knowledge-assistant
\`\`\`

### 2. Create a virtual environment
\`\`\`bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
\`\`\`

### 3. Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Install Ollama + pull a model
\`\`\`bash
# Install: https://ollama.com/download
ollama pull llama3
\`\`\`

### 5. Configure environment
\`\`\`bash
cp .env.example .env
# Set OLLAMA_MODEL=llama3 (or any model you pulled)
\`\`\`

### 6. Run
\`\`\`bash
streamlit run app.py
\`\`\`

Open **http://localhost:8501**

---

## 🔍 How RAG Works Here
```
Upload Document
    → Extract text (page + line numbers)
    → Split into overlapping chunks (400 chars, 80 overlap)
    → Embed with BAAI/bge-small-en-v1.5
    → Store in FAISS index

Ask a Question
    → Embed question
    → FAISS similarity search → Top 5 chunks
    → Send chunks + question to Ollama
    → Return answer with citations + confidence score
```

---

## 🔒 No-Hallucination Policy

The system prompt strictly instructs the model:
```
Answer ONLY using the provided document context.
If the answer is not present, reply with exactly: Not found
Do not use any external knowledge. Do not guess or infer.
```

---

## ⚙️ Supported Ollama Models

| Model | Command |
|---|---|
| Llama 3 | `ollama pull llama3` |
| Mistral | `ollama pull mistral` |
| Phi-3 | `ollama pull phi3` |
| Gemma | `ollama pull gemma` |
| DeepSeek-R1 | `ollama pull deepseek-r1` |

Change model in `.env`: `OLLAMA_MODEL=mistral`

---

## 📄 License

MIT License — free to use, modify, and distribute.
```
