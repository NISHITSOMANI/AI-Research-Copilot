# ingest.py
import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Config
DATA_DIR = "data"
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "metadata.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200     # words per chunk (tunable)
CHUNK_OVERLAP = 40   # overlap words (tunable)

# Load embedder once
embedder = SentenceTransformer(EMBEDDING_MODEL)


def load_text_files(data_dir=DATA_DIR):
    """Load text files and simple txt content from pdfs."""
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if os.path.isdir(fpath):
            continue
        if fname.lower().endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                docs.append({"source": fname, "page": None, "text": text})
        elif fname.lower().endswith(".pdf"):
            try:
                reader = PdfReader(fpath)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        docs.append({"source": fname, "page": i+1, "text": text.strip()})
            except Exception as e:
                print(f"[ingest] Could not read PDF {fname}: {e}")
    return docs


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
    return chunks


def build_index(data_dir=DATA_DIR, index_dir=INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)

    docs = load_text_files(data_dir)
    texts = []
    metadatas = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for c_id, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "source": doc["source"],
                "page": doc.get("page"),
                "chunk_id": c_id
            })

    if not texts:
        print("[ingest] No documents found in data/ — add .txt or .pdf files first.")
        return

    print(f"[ingest] Creating embeddings for {len(texts)} chunks using {EMBEDDING_MODEL} ...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Normalize for cosine similarity using inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    print(f"[ingest] Building FAISS index (dim={dim}) ...")
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings.astype('float32'))

    # Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metadatas": metadatas}, f, ensure_ascii=False, indent=2)

    print(f"[ingest] Done. Index saved to {INDEX_FILE}, metadata to {META_FILE}.")


if __name__ == "__main__":
    build_index()
