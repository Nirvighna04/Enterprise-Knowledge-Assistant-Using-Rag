"""
rag.py — Full RAG pipeline:
    1. Document ingestion (PDF / DOCX / TXT) with page + line metadata
    2. Overlapping chunking
    3. Embeddings via sentence-transformers (BAAI/bge-small-en-v1.5)
    4. FAISS vector store (persistent)
    5. Retrieval + LLM call via Ollama (chat-ollama)
    6. Strict no-hallucination: returns "Not found" when answer absent
"""

import os
import json
import pickle
import numpy as np
import faiss

from typing import List, Dict, Tuple

# ── Sentence-Transformers ─────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Document parsers ──────────────────────────────────────────────────────────
import pdfplumber
from docx import Document as DocxDocument

# ── Ollama ────────────────────────────────────────────────────────────────────
import ollama

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR  = os.path.join(BASE_DIR, "vector_store")
DATA_DIR    = os.path.join(BASE_DIR, "data")
INDEX_PATH  = os.path.join(VECTOR_DIR, "faiss.index")
META_PATH   = os.path.join(VECTOR_DIR, "metadata.pkl")

os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"   # fast + accurate
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3")  # change to any local model
CHUNK_SIZE    = 400    # characters per chunk
CHUNK_OVERLAP = 80     # character overlap between chunks
TOP_K         = 5      # number of chunks to retrieve

# ── Load embedding model (cached globally) ────────────────────────────────────
_embedder = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        try:
            _embedder = SentenceTransformer(EMBED_MODEL)
        except Exception:
            _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


# ── 1. Document ingestion ─────────────────────────────────────────────────────

def extract_text(file_path: str) -> List[Dict]:
    """
    Returns a list of dicts:
        { "text": str, "page": int, "line": int, "doc_name": str }
    """
    ext      = os.path.splitext(file_path)[1].lower()
    doc_name = os.path.basename(file_path)
    records  = []

    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                for line_num, line in enumerate(raw.splitlines(), start=1):
                    if line.strip():
                        records.append({
                            "text":     line.strip(),
                            "page":     page_num,
                            "line":     line_num,
                            "doc_name": doc_name,
                        })

    elif ext in (".docx", ".doc"):
        doc = DocxDocument(file_path)
        line_num = 1
        for para_num, para in enumerate(doc.paragraphs, start=1):
            if para.text.strip():
                records.append({
                    "text":     para.text.strip(),
                    "page":     1,          # DOCX has no true page numbers
                    "line":     line_num,
                    "doc_name": doc_name,
                })
                line_num += 1

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, start=1):
                if line.strip():
                    records.append({
                        "text":     line.strip(),
                        "page":     1,
                        "line":     line_num,
                        "doc_name": doc_name,
                    })
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return records


# ── 2. Chunking ───────────────────────────────────────────────────────────────

def chunk_records(records: List[Dict]) -> List[Dict]:
    """
    Merge lines into overlapping text chunks while preserving
    the page/line of the FIRST line in each chunk.
    """
    # Concatenate all text with separators, keeping position map
    full_text = ""
    positions = []   # list of (page, line) for each char position
    for rec in records:
        start = len(full_text)
        full_text += rec["text"] + " "
        positions.extend([(rec["page"], rec["line"])] * (len(rec["text"]) + 1))

    chunks = []
    start  = 0
    while start < len(full_text):
        end   = min(start + CHUNK_SIZE, len(full_text))
        chunk = full_text[start:end].strip()
        if chunk:
            page, line = positions[start]
            chunks.append({
                "text":     chunk,
                "page":     page,
                "line":     line,
                "doc_name": records[0]["doc_name"] if records else "unknown",
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ── 3 & 4. Embed + Store in FAISS ────────────────────────────────────────────

def load_index() -> Tuple[faiss.Index | None, List[Dict]]:
    """Load existing FAISS index and metadata from disk."""
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index    = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, []


def save_index(index: faiss.Index, metadata: List[Dict]):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def ingest_document(file_path: str) -> int:
    """
    Full pipeline: extract → chunk → embed → add to FAISS.
    Returns number of chunks added.
    """
    embedder = get_embedder()
    records  = extract_text(file_path)
    chunks   = chunk_records(records)

    if not chunks:
        return 0

    texts      = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True).astype("float32")

    dim = embeddings.shape[1]

    # Load or create index
    index, metadata = load_index()
    if index is None:
        index = faiss.IndexFlatIP(dim)   # Inner Product = cosine on normalised vecs

    index.add(embeddings)
    metadata.extend(chunks)
    save_index(index, metadata)

    return len(chunks)


# ── 5. Retrieval ──────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Returns top_k most relevant chunks with similarity scores.
    Each item: { text, page, line, doc_name, score }
    """
    index, metadata = load_index()
    if index is None or index.ntotal == 0:
        return []

    embedder = get_embedder()
    q_vec    = embedder.encode([query], normalize_embeddings=True).astype("float32")

    scores, indices = index.search(q_vec, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def confidence_level(chunks: List[Dict]) -> str:
    """Derive Low/Medium/High confidence from scores and chunk count."""
    if not chunks:
        return "Low"
    avg_score = sum(c["score"] for c in chunks) / len(chunks)
    if avg_score > 0.75 and len(chunks) >= 3:
        return "High"
    if avg_score > 0.50:
        return "Medium"
    return "Low"


# ── 6. LLM call via Ollama ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a company knowledge assistant for the {department} department and {job_role} role.
Answer ONLY using the provided document context below.
If the answer is not present in the context, reply with exactly: Not found
Do not use any external knowledge. Do not guess or infer beyond the context.
Always cite your sources (document name, page, line)."""

USER_PROMPT = """Context:
{context}

User Question:
{question}

Provide:
1. Final Answer (or "Not found")
2. Citations: document name, page number, line number for each source used
3. Confidence: Low / Medium / High
4. 2–3 Suggested Follow-up Questions (skip if answer is "Not found")
"""


def ask_llm(
    question:   str,
    department: str,
    job_role:   str,
    chunks:     List[Dict],
) -> str:
    """
    Call Ollama with retrieved context.
    Returns the model's raw text response.
    """
    context = "\n\n".join(
        f"[Doc: {c['doc_name']} | Page: {c['page']} | Line: {c['line']}]\n{c['text']}"
        for c in chunks
    )

    system = SYSTEM_PROMPT.format(department=department, job_role=job_role)
    user   = USER_PROMPT.format(context=context, question=question)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
    )
    return response["message"]["content"]


# ── Main RAG function ─────────────────────────────────────────────────────────

def rag_query(
    question:   str,
    department: str = "General",
    job_role:   str = "Employee",
) -> Dict:
    """
    Full RAG query. Returns dict with keys:
        answer, citations, confidence, follow_ups, chunks
    """
    chunks = retrieve(question)

    if not chunks:
        return {
            "answer":     "Not found",
            "citations":  [],
            "confidence": "Low",
            "follow_ups": [],
            "chunks":     [],
        }

    raw    = ask_llm(question, department, job_role, chunks)
    conf   = confidence_level(chunks)

    citations = [
        {
            "doc_name":  c["doc_name"],
            "page":      c["page"],
            "line":      c["line"],
            "file_path": os.path.join(DATA_DIR, c["doc_name"]),
        }
        for c in chunks
    ]

    return {
        "answer":     raw,
        "citations":  citations,
        "confidence": conf,
        "follow_ups": [],   # Follow-ups are embedded in the LLM answer text
        "chunks":     chunks,
    }
