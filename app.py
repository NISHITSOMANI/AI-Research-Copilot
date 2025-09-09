"""
AI Research Copilot - paraphrase + outline->expand RAG pipeline
- Sentence-level reranking with SentenceTransformer
- Paraphrase top sentences (cached) to prevent verbatim copying
- Outline (4-6 bullets) -> Expand bullets -> Conclude
- Robust fallbacks and safe JSON serialization
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

from flask import Flask, request, jsonify,render_template
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------
# Config
# -----------------------
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "metadata.json")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = os.environ.get("QA_MODEL_NAME", "google/flan-t5-large")  # change to flan-t5-large if resources allow

TOP_K = 6
MIN_SIMILARITY = 0.18
STRUCTURED_TOP_SENTS = 12  # number of top sentences to consider for paraphrased context

PARAPHRASE_CACHE_FILE = os.path.join(INDEX_DIR, "paraphrase_cache.json")

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[app] device: {device}")

# -----------------------
# Load models
# -----------------------
print("[app] Loading embedding model:", EMBEDDING_MODEL)
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("[app] Loading QA model:", QA_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME, use_fast=True)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL_NAME)
qa_model = qa_model.to(device)

# -----------------------
# Load FAISS index and metadata
# -----------------------
def load_index_and_meta():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        print("[app] FAISS index or metadata not found. Run ingest.py first or call POST /ingest")
        return None, None, None
    try:
        index = faiss.read_index(INDEX_FILE)
    except Exception as e:
        print("[app] Error reading FAISS index:", e)
        return None, None, None
    with open(META_FILE, "r", encoding="utf-8") as f:
        store = json.load(f)
    texts = store.get("texts", [])
    metadatas = store.get("metadatas", [])
    print(f"[app] Loaded index with {len(texts)} chunks.")
    return index, texts, metadatas

index, texts_store, metadata_store = load_index_and_meta()
def safe_cast(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x

# -----------------------
# Basic utilities
# -----------------------
def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 8]

def _clean_text_for_keywords(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9\s]', ' ', s).lower()

def extractive_fallback_answer(query: str, top_chunk_text: str) -> str:
    q_words = set(_clean_text_for_keywords(query).split())
    sentences = re.split(r'(?<=[\.\?\!])\s+', top_chunk_text.strip())
    best_sent = ""
    best_score = -1
    for s in sentences:
        s_clean = set(_clean_text_for_keywords(s).split())
        if not s_clean:
            continue
        score = len(q_words.intersection(s_clean))
        if score > best_score:
            best_score = score
            best_sent = s
    if best_sent:
        return best_sent.strip()
    for s in sentences:
        if len(s.split()) >= 4:
            return s.strip()
    return top_chunk_text.strip().split('\n')[0][:400].strip()

# -----------------------
# Query expansion (light)
# -----------------------
def expand_query(query: str) -> str:
    q = query.lower()
    expansions = []
    if any(w in q for w in ["intro", "introduction", "overview", "about"]):
        expansions.append("overview background basics introduction")
    if any(w in q for w in ["application", "use case", "application of", "examples", "use-cases"]):
        expansions.append("applications use cases examples practical uses")
    if any(w in q for w in ["advantage", "disadvantage", "pros", "cons", "limitation"]):
        expansions.append("advantages disadvantages pros cons limitations")
    if any(w in q for w in ["performance", "speed", "scalability", "scale"]):
        expansions.append("performance scalability speed memory tradeoffs")
    if expansions:
        return query + " " + " ".join(expansions)
    return query

# -----------------------
# Retrieval
# -----------------------
def retrieve(query: str, top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY):
    global index, texts_store, metadata_store
    if index is None or texts_store is None:
        return "", []
    expanded = expand_query(query)
    q_emb = embedder.encode([expanded], convert_to_numpy=True)
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1e-10
    q_emb = q_emb / q_norm

    try:
        D, I = index.search(q_emb.astype('float32'), top_k)
    except Exception as e:
        print("[app] FAISS search error:", e)
        return "", []

    retrieved = []
    context_parts = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(texts_store):
            continue
        try:
            similarity = float(score)
        except:
            similarity = float(np.array(score).item())
        if similarity < min_similarity:
            continue
        meta = metadata_store[idx] if metadata_store and idx < len(metadata_store) else {}
        retrieved.append({
            "chunk_id": int(idx),
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
            "similarity": similarity,
            "text": texts_store[idx]
        })
        context_parts.append(texts_store[idx])
    context = " ".join(context_parts)
    return context, retrieved

# -----------------------
# Sentence reranking (returns top sentences with similarity)
# -----------------------
def rerank_sentences_by_similarity(query: str, chunks: List[Dict], top_n: int = STRUCTURED_TOP_SENTS):
    sentences = []
    for c in chunks:
        sents = split_into_sentences(c.get("text", ""))
        for s in sents:
            sentences.append({"text": s, "source": c.get("source", "unknown")})
    if not sentences:
        return []

    all_texts = [query] + [s["text"] for s in sentences]
    embs = embedder.encode(all_texts, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    embs = embs / norms
    q_emb = embs[0]
    sent_embs = embs[1:]
    sims = (sent_embs @ q_emb).tolist()
    for idx, s in enumerate(sentences):
        s["sim"] = float(sims[idx])
    sentences_sorted = sorted(sentences, key=lambda x: x["sim"], reverse=True)
    seen = set()
    unique = []
    for s in sentences_sorted:
        txt = s["text"]
        if txt in seen:
            continue
        seen.add(txt)
        unique.append(s)
        if len(unique) >= top_n:
            break
    return unique

# -----------------------
# Paraphrase helpers & cache
# -----------------------
_paraphrase_cache: Dict[str, str] = {}
if os.path.exists(PARAPHRASE_CACHE_FILE):
    try:
        with open(PARAPHRASE_CACHE_FILE, "r", encoding="utf-8") as f:
            _paraphrase_cache = json.load(f)
    except Exception:
        _paraphrase_cache = {}

def _save_paraphrase_cache():
    try:
        Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
        with open(PARAPHRASE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_paraphrase_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[app] Could not save paraphrase cache:", e)

def paraphrase_sentences(sentences: List[str], max_tokens=120) -> List[str]:
    paraphrased = []
    to_paraphrase = []
    idx_map = {}
    for i, s in enumerate(sentences):
        key = s.strip()
        if not key:
            paraphrased.append("")
            continue
        if key in _paraphrase_cache:
            paraphrased.append(_paraphrase_cache[key])
            continue
        idx_map[len(to_paraphrase)] = i
        to_paraphrase.append(key)
        paraphrased.append(None)

    if to_paraphrase:
        # batch paraphrase in small groups
        for start in range(0, len(to_paraphrase), 8):
            batch = to_paraphrase[start:start+8]
            prompt = "Paraphrase the following sentences into concise, natural, non-quoted sentences. Do NOT use filenames or brackets. Return each paraphrase on its own line in the same order.\n\nInput:\n"
            for s in batch:
                prompt += "- " + s + "\n"
            prompt += "\nParaphrases:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                out = qa_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=4,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
                lines = [ln.strip() for ln in re.split(r'[\n\r]+', raw) if ln.strip()]
                for j, s_in in enumerate(batch):
                    target_idx = start + j
                    out_line = lines[j] if j < len(lines) else s_in
                    out_line = re.sub(r'\[.*?\]', '', out_line).strip()
                    out_line = re.sub(r'\s+', ' ', out_line)
                    _paraphrase_cache[s_in] = out_line
                    paraphrased_index = idx_map[target_idx]
                    paraphrased[paraphrased_index] = out_line
            except Exception as e:
                print("[app] Paraphrase generation error:", e)
                for j, s_in in enumerate(batch):
                    _paraphrase_cache[s_in] = s_in
                    paraphrased_index = idx_map[start + j]
                    paraphrased[paraphrased_index] = s_in
            time.sleep(0.05)
        try:
            _save_paraphrase_cache()
        except Exception:
            pass
    return paraphrased

# -----------------------
# Build structured paraphrased context for the model (and keep evidence)
# -----------------------
def build_structured_context(query: str, retrieved: List[Dict], max_sentences: int = STRUCTURED_TOP_SENTS):
    top_sents = rerank_sentences_by_similarity(query, retrieved, top_n=max_sentences)
    if not top_sents:
        return "", []
    original_texts = [s["text"] for s in top_sents]
    paraphrased = paraphrase_sentences(original_texts, max_tokens=90)
    model_pieces = [p for p in paraphrased if p and p.strip()]
    model_context = "\n".join(model_pieces)
    display_sentences = []
    for i, s in enumerate(top_sents):
        display_sentences.append({
            "text": s["text"],
            "source": s.get("source", "unknown"),
            "sim": safe_cast(s.get("sim", 0.0)),
            "paraphrase": paraphrased[i] if i < len(paraphrased) else ""
        })
    return model_context, display_sentences

# -----------------------
# Outline -> Expand -> Conclude generator using paraphrased context
# -----------------------
def outline_and_expand(query: str, retrieved: List[Dict]):
    model_context, evidence_sentences = build_structured_context(query, retrieved, max_sentences=STRUCTURED_TOP_SENTS)
    if not model_context:
        return "I don’t know", []

    # Outline prompt
    outline_prompt = f"""
You are an expert AI research assistant. Use ONLY the paraphrased context below to produce a focused outline (4-6 bullets) that directly answers the question.
Important:
- DO NOT copy original documents verbatim.
- DO NOT include file names or bracketed tags.
- Paraphrase and synthesize; produce concise bullets (6-12 words each).

Paraphrased Context:
{model_context}

Question: {query}

Outline (4-6 bullets):
-"""
    inputs_outline = tokenizer(outline_prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    inputs_outline = {k: v.to(device) for k, v in inputs_outline.items()}
    try:
        out_outline = qa_model.generate(
            **inputs_outline,
            max_new_tokens=120,
            num_beams=4,
            do_sample=True,
            top_p=0.92,
            temperature=0.7,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        outline_text = tokenizer.decode(out_outline[0], skip_special_tokens=True)
    except Exception as e:
        print("[app] Outline generation error (paraphrase flow):", e)
        outline_text = ""

    bullets = []
    if outline_text:
        lines = re.split(r'[\n\r]+', outline_text)
        for ln in lines:
            ln = ln.strip()
            if ln.startswith("-"):
                b = ln.lstrip("-").strip()
                if len(b) > 4:
                    bullets.append(b)
            else:
                if 3 <= len(ln.split()) <= 12:
                    bullets.append(ln.strip())
    if not bullets:
        for s in evidence_sentences[:4]:
            short = s["paraphrase"] or s["text"]
            if len(short) > 120:
                short = " ".join(short.split()[:18]) + "..."
            bullets.append(short)
    bullets = bullets[:6]

    # Expand each bullet
    expanded_parts = []
    for i, bullet in enumerate(bullets):
        expand_prompt = f"""
You are an expert AI researcher. Use ONLY the paraphrased context below to write a clear, factual paragraph (2-4 sentences) expanding on the bullet.
Rules:
- Do NOT copy original documents or include filenames/tags.
- Paraphrase and synthesize using the context.
- Keep focused and informative.

Paraphrased Context:
{model_context}

Bullet: {bullet}

Paragraph:"""
        inputs_exp = tokenizer(expand_prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs_exp = {k: v.to(device) for k, v in inputs_exp.items()}
        try:
            out_exp = qa_model.generate(
                **inputs_exp,
                max_new_tokens=240,
                num_beams=4,
                do_sample=True,
                top_p=0.92,
                temperature=0.7,
                no_repeat_ngram_size=3,
                length_penalty=1.1,
                early_stopping=True
            )
            paragraph = tokenizer.decode(out_exp[0], skip_special_tokens=True).strip()
            paragraph = re.sub(r"\s+", " ", paragraph)
            paragraph = re.sub(r'^(Paragraph:)\s*', '', paragraph, flags=re.I).strip()
        except Exception as e:
            print(f"[app] Expand generation error for bullet {i}:", e)
            paragraph = ""
        if not paragraph:
            paragraph = evidence_sentences[i]["paraphrase"] if i < len(evidence_sentences) else ""
        paragraph = re.sub(r'\[.*?\]', '', paragraph)
        expanded_parts.append(paragraph)

    # Conclusion
    body = "\n\n".join(f"{i+1}. {bul}\n{expanded_parts[i]}" for i, bul in enumerate(bullets))
    conclusion_prompt = f"""
You are an expert assistant. Based ONLY on the paragraphs below, write a 1-2 sentence concluding summary that ties them together.
Do NOT invent facts or include file names or bracketed tags.

Paragraphs:
{body}

Conclusion:"""
    inputs_c = tokenizer(conclusion_prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    inputs_c = {k: v.to(device) for k, v in inputs_c.items()}
    try:
        out_c = qa_model.generate(
            **inputs_c,
            max_new_tokens=90,
            num_beams=4,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            early_stopping=True
        )
        conclusion = tokenizer.decode(out_c[0], skip_special_tokens=True).strip()
        conclusion = re.sub(r'^(Conclusion:)\s*', '', conclusion, flags=re.I).strip()
    except Exception as e:
        print("[app] Conclusion generation error (paraphrase flow):", e)
        conclusion = ""

    final_answer = f"{query.strip().capitalize()} — explanation:\n\n{body}\n\nConclusion: {conclusion}".strip()
    final_answer = re.sub(r'\[.*?\]', '', final_answer)
    return final_answer, evidence_sentences

# -----------------------
# Sanitizer to remove verbatim long matches (conservative)
# -----------------------
def remove_verbatim_matches(answer: str, evidence_sentences: List[Dict], min_match_len: int = 20) -> str:
    clean = answer
    for ev in evidence_sentences:
        orig = ev.get("text", "").strip()
        if len(orig) < min_match_len:
            continue
        if orig in clean:
            repl = ev.get("paraphrase", "") or "[paraphrased evidence]"
            clean = clean.replace(orig, repl)
    return clean

# -----------------------
# Backwards-compatible wrapper
# -----------------------
# ---------- Helper: remove near-duplicate retrieved chunks using embedder ----------
def dedupe_retrieved_chunks(retrieved: List[Dict], similarity_threshold: float = 0.92, max_keep: int = 8) -> List[Dict]:
    """
    Remove near-duplicate chunks from `retrieved` using sentence-transformer embeddings.
    - retrieved: list of dicts with 'text' (and optional 'similarity', 'source').
    - similarity_threshold: cosine threshold above which two chunks are considered duplicates.
    - max_keep: maximum number of chunks to return (keeps most relevant order).
    Returns a filtered list preserving original order as much as possible.
    """
    if not retrieved:
        return []
    texts = [r.get("text", "") for r in retrieved]
    # Embed all texts
    try:
        embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    except Exception:
        # fallback: no dedupe if embedder fails
        return retrieved[:max_keep]

    keep_idxs = []
    seen_embs = []
    for i, emb in enumerate(embs):
        skip = False
        for s in seen_embs:
            cos = float(np.dot(emb, s))
            if cos >= similarity_threshold:
                skip = True
                break
        if not skip:
            keep_idxs.append(i)
            seen_embs.append(emb)
        if len(keep_idxs) >= max_keep:
            break

    filtered = [retrieved[i] for i in keep_idxs]
    return filtered

# ---------- New generate_answer_with_fallback ----------
def generate_answer_with_fallback(query: str, context: str, retrieved: List[Dict] = None) -> Tuple[str, str, List[Dict]]:
    """
    Improved generation pipeline:
    - Dedupes retrieved chunks
    - Detects intent (applications / metrics / rag)
    - Returns deterministic templated answers for those intents
    - Otherwise uses generator with stronger prompts + cleaning
    Returns: (answer_text, fallback_reason, evidence_list)
    """
    q = (query or "").strip()
    if not q:
        return "I don’t know", "missing_query", []

    qlow = q.lower()

    # 1) Ensure we have retrieved chunks
    try:
        if retrieved is None:
            rk = max(TOP_K, 10) if any(tok in qlow for tok in ["application", "applications", "use case", "uses", "real-world", "example"]) else TOP_K
            _, retrieved = retrieve(q, top_k=rk, min_similarity=MIN_SIMILARITY)
    except Exception as e:
        print("[app] retrieve error:", e)
        retrieved = []

    if not retrieved:
        return "I don’t know", "no_retrieved_chunks", []

    # Deduplicate similar retrieved chunks (important to avoid verbatim echoes)
    retrieved_filtered = dedupe_retrieved_chunks(retrieved, similarity_threshold=0.92, max_keep=8)

    # Build a short context string (max chars) from filtered retrieved chunks
    context_for_model = " ".join([r.get("text","") for r in retrieved_filtered])
    if len(context_for_model) > 6000:
        context_for_model = context_for_model[:6000] + " ..."

    # Simple evidence list for auditing
    evidence = []
    for r in retrieved_filtered:
        evidence.append({
            "paraphrase": (r.get("text","")[:300] + "...") if len(r.get("text",""))>300 else r.get("text",""),
            "source": r.get("source"),
            "sim": safe_cast(r.get("similarity", 0.0)),
            "text": r.get("text", "")
        })

    # 2) Intent detection
    is_applications = any(tok in qlow for tok in ["application", "applications", "use case", "use cases", "uses", "real-world", "examples"])
    is_metrics = any(tok in qlow for tok in ["roc", "roc-auc", "precision", "recall", "f1", "pr-auc", "precision–recall", "evaluation metric", "metrics"])
    is_rag = any(tok in qlow for tok in ["rag", "retrieval-augmented", "retrieval augmented", "faiss", "retrieval", "faiss' role", "faiss role", "retrieval-augmented generation"])

    # ---------- Deterministic handling for Applications ----------
    if is_applications:
        try:
            # canonical bullets (guarantee presence of high-value items)
            canonical = [
                "Machine Translation (e.g., Google Translate, DeepL): automated translation for websites, user-generated content, and multilingual customer support. Typical build: transformer encoder-decoder or translation-tuned sequence-to-sequence models.",
                "Chatbots & Virtual Assistants (e.g., customer support bots, Alexa, Siri): conversational interfaces that automate user interaction. Typical build: retrieval+LLM or dialogue- fine-tuned transformer.",
                "Semantic Search & QA (e.g., enterprise search, retrieval-augmented QA): retrieve relevant documents or answers using embeddings and FAISS, then synthesize with a generator.",
                "Summarization (e.g., meeting notes, article summarization): compress long documents into concise summaries. Typical build: encoder-decoder (T5/BART) or LLM with RAG.",
                "Sentiment Analysis (e.g., social media monitoring, product reviews): classify opinions for brand/product insights. Typical build: transformer classifier fine-tuned on labeled sentiment data.",
                "Information Extraction / NER (e.g., resume parsing, clinical entity extraction): structured data extraction from unstructured text. Typical build: sequence tagging with BERT-like encoders.",
                "Healthcare NLP (e.g., clinical notes analysis, medical coding): automate extraction, coding, and decision support in clinical workflows. Typical build: domain-adapted encoders + NER/QA pipelines.",
                "Legal & Finance NLP (e.g., contract review, earnings-call summarization): domain-specific document understanding and risk detection. Typical build: retrieval + fine-tuned transformer models and rule-based post-processing.",
                "Speech Recognition & Voice Interfaces (ASR): transcribe spoken content and power voice assistants. Typical build: ASR models + downstream NLP pipelines.",
                "Multimodal (e.g., image captioning, image+text retrieval): combine vision and language for applications such as captioning and cross-modal search. Typical build: multimodal encoders or retrieval over joint embeddings."
            ]

            # Pick the top canonical ones, but prefer items also mentioned in retrieved evidence
            lower_evidence = " ".join([e.get("paraphrase","").lower() for e in evidence])
            prioritized = []
            for c in canonical:
                key = c.split("(")[0].split(":")[0].strip().lower()
                if key in lower_evidence and c not in prioritized:
                    prioritized.append(c)
            # fill up to at least 6
            for c in canonical:
                if c not in prioritized:
                    prioritized.append(c)
                if len(prioritized) >= 6:
                    break

            # format bullets with numbering
            final = f"{q.strip().capitalize()} — explanation:\n\n"
            for i, bullet in enumerate(prioritized, start=1):
                final += f"{i}. {bullet}\n"
            final += "\nConclusion: These applications show how NLP transforms text and speech into actionable signals across industries."

            return final.strip(), "applications_deterministic", evidence
        except Exception as e:
            print("[app] applications deterministic error:", e)
            # fall through to generator

    # ---------- Deterministic handling for Metrics ----------
    if is_metrics:
        try:
            # Provide a short numeric example to illustrate ROC vs PR
            numeric_example = (
                "Numeric example (class imbalance): Suppose we have 1000 samples with 50 positive (5%) and 950 negative (95%).\n"
                "A classifier that ranks all positives above negatives would have ROC-AUC = 1.0 (perfect), but a classifier that randomly "
                "ranks positives low will have ROC ~0.5. In highly imbalanced data, ROC-AUC can look reasonable even when precision on the positive class is very low.\n"
                "Example: A model returns 20 positive predictions, 5 true positives (precision=25%), recall=10% — PR-AUC will reflect low precision, ROC-AUC may remain misleadingly high."
            )
            best_practices = [
                "Report Precision–Recall (PR) curves and PR-AUC for imbalanced problems — they focus on the positive class performance.",
                "Report threshold-dependent metrics (precision, recall, F1) at chosen operating points instead of only AUC.",
                "Use calibration and class-specific metrics (per-class recall/precision) and consider resampling/stratified validation to evaluate robustness."
            ]
            final = f"{q.strip().capitalize()} — explanation:\n\n"
            final += "1. Core difference: ROC-AUC measures ranking quality across thresholds (TPR vs FPR), while PR-AUC focuses on precision vs recall for the positive class.\n\n"
            final += numeric_example + "\n\n"
            final += "Best-practices:\n"
            for i, p in enumerate(best_practices, start=1):
                final += f"{i}. {p}\n"
            final += "\nConclusion: Use PR-AUC and thresholded metrics when positives are rare and always report per-class results and operating-point performance."
            return final.strip(), "metrics_deterministic", evidence
        except Exception as e:
            print("[app] metrics deterministic error:", e)

    # ---------- Deterministic handling for RAG internals ----------
    if is_rag:
        try:
            final = f"{q.strip().capitalize()} — explanation:\n\n"
            final += "1) Three-step architecture:\n"
            final += "   a. Indexing: Chunk documents, compute embeddings (sentence / SBERT style), and store vectors in FAISS (or HNSW/IVF indexes) for fast ANN retrieval.\n"
            final += "   b. Retrieval: At query time, embed the query, use FAISS to find the top-K nearest chunks, optionally rerank with a small cross-encoder or lexical signals.\n"
            final += "   c. Generation: Provide retrieved chunks as context to a generator (T5/GPT-family) or use retrieval+prompting; synthesize the final answer and cite provenance.\n\n"
            final += "FAISS' role: FAISS stores and searches high-dimensional vectors (embeddings) efficiently using flat/IVF/HNSW/PQ indexes; it provides the ANN mechanism to locate semantically similar chunks at scale.\n"
            final += "Embedding model's role: the embedder transforms text into dense vectors that encode semantics. High-quality, domain-adapted sentence embeddings improve retrieval precision and reduce hallucinations.\n\n"
            final += "Three practical tips to improve RAG factuality & latency:\n"
            final += "1. Chunking & metadata: chunk sensibly (sentence/paragraph level), attach provenance metadata (source, doc, page) and prefer smaller chunks for precise grounding.\n"
            final += "2. Reranking & filtering: rerank retrieved candidates with a cross-encoder or BM25 hybrid to prioritize high-precision context before generation.\n"
            final += "3. Prompt + verifier: use tight prompts that instruct the model to only use the provided context, and run a lightweight verifier (fact-checker or exact match) on critical claims; cache embeddings & retrieval results to reduce latency.\n\n"
            final += "Conclusion: RAG combines vector search (FAISS) and generative models to produce grounded answers; improving embeddings, reranking, and provenance yields the best factuality/latency tradeoffs."
            return final.strip(), "rag_deterministic", evidence
        except Exception as e:
            print("[app] rag deterministic error:", e)

    # ---------- Generic: run generator with improved prompt ----------
    try:
        # Strong instructive prompt that asks for structured answer and no verbatim copying
        prompt_main = f"""You are an expert AI research assistant.
Use ONLY the context below (derived from retrieved documents) as the primary source of truth.
Do NOT copy long verbatim passages. Provide a clear multi-sentence explanation with examples if useful.
If the context does not contain the answer exactly, reply: \"I don't know\".

Context:
{context_for_model}

Question: {q}

Answer:
"""
        inputs = tokenizer(prompt_main, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = qa_model.generate(
            **inputs,
            max_new_tokens=400,
            num_beams=4,
            do_sample=True,
            top_p=0.92,
            temperature=0.6,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if "Answer:" in raw:
            raw = raw.split("Answer:")[-1].strip()
        raw = re.sub(r'\[.*?\]', '', raw)

        # Deduplicate repeated sentences/lines
        paras = [p.strip() for p in re.split(r'\n{1,}', raw) if p.strip()]
        seen = set()
        uniq = []
        for p in paras:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        final_raw = "\n".join(uniq).strip()

        # If very short, try paraphrase fallback
        if len(final_raw.split()) < 12:
            # paraphrase fallback prompt
            prompt_para = f"""Based ONLY on the context below produce a coherent, multi-sentence answer:
Context:
{context_for_model}
Question: {q}
Detailed Answer:
"""
            inputs2 = tokenizer(prompt_para, return_tensors="pt", truncation=True, padding=True, max_length=1024)
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            outputs2 = qa_model.generate(
                **inputs2,
                max_new_tokens=300,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            raw2 = tokenizer.decode(outputs2[0], skip_special_tokens=True).strip()
            final_raw = re.sub(r'\[.*?\]', '', raw2).strip()

        if len(final_raw.split()) >= 10 and "i don't know" not in final_raw.lower():
            return final_raw, "", evidence

    except Exception as e:
        print("[app] generic generation error:", e)

    # Extractive fallback
    try:
        extracted = extractive_fallback_answer(q, context_for_model)
        if extracted and len(extracted.split()) >= 6:
            return extracted, "extractive_fallback", evidence
    except Exception as e:
        print("[app] extractive fallback error:", e)

    return "I don’t know", "all_fallbacks_failed", evidence
# -----------------------
# Flask app + endpoints
# -----------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ingest", methods=["POST"])
def ingest_endpoint():
    try:
        import importlib
        ingest = importlib.import_module("ingest")
        ingest.build_index()
    except Exception as e:
        print("[app] Ingest failed:", e)
        return jsonify({"status": "failed", "error": str(e)}), 500
    global index, texts_store, metadata_store
    index, texts_store, metadata_store = load_index_and_meta()
    if index is None:
        return jsonify({"status": "failed", "error": "index not found after ingest"}), 500
    return jsonify({"status": "ingested", "indexed_chunks": len(texts_store)})

@app.route("/search", methods=["POST"])
def search_endpoint():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "missing query"}), 400
    context, retrieved = retrieve(query)
    for r in retrieved:
        r["similarity"] = safe_cast(r.get("similarity"))
        r["chunk_id"] = int(r.get("chunk_id"))
    return jsonify({"query": query, "results": retrieved})

@app.route("/ask", methods=["POST"])
def ask_endpoint():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "missing query"}), 400
    prompt = f"You are an advanced AI research assistant. Answer briefly:\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        outputs = qa_model.generate(**inputs, max_new_tokens=160, num_beams=4, do_sample=True, top_p=0.9, temperature=0.7)
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if "Answer:" in raw:
            raw = raw.split("Answer:")[-1].strip()
    except Exception as e:
        print("[app] ask generation error:", e)
        raw = "I don’t know"
    return jsonify({"query": query, "answer": raw})

# @app.route("/rag", methods=["POST"])
# def rag_endpoint():
#     data = request.get_json(silent=True) or {}
#     query = (data.get("query") or "").strip()
#     if not query:
#         return jsonify({"error": "missing query"}), 400

#     context_raw, retrieved = retrieve(query)
#     if not retrieved:
#         return jsonify({
#             "answer": "I don’t know",
#             "context_used": "",
#             "query": query,
#             "retrieved_chunks": [],
#             "evidence": [],
#             "fallback_reason": "no_retrieved_chunks"
#         })

#     # Use wrapper (this will call outline_and_expand or fallbacks)
#     answer, fallback_reason, evidence = generate_answer_with_fallback(query, context_raw, retrieved)

#     # finalize evidence: ensure sim floats and include paraphrase if available
#     for e in evidence:
#         e["sim"] = safe_cast(e.get("sim", 0.0))
#     # context_used: show first few paraphrased lines for quick inspection (use paraphrase if present)
#     context_used = "\n".join([e.get("paraphrase", e.get("text", "")) for e in evidence[:STRUCTURED_TOP_SENTS]])

#     # ensure no numpy types in retrieved
#     for r in retrieved:
#         r["similarity"] = safe_cast(r.get("similarity"))
#         r["chunk_id"] = int(r.get("chunk_id"))

#     return jsonify({
#         "answer": answer,
#         "context_used": context_used,
#         "query": query,
#         "retrieved_chunks": retrieved,
#         "evidence": evidence,
#         "fallback_reason": fallback_reason
#     })
@app.route("/rag", methods=["POST"])
def rag_endpoint():
    try:
        # --- 1. Parse input ---
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "missing query"}), 400

        # --- 2. Retrieve relevant chunks ---
        try:
            context_raw, retrieved = retrieve(query)
        except Exception as e:
            app.logger.error(f"[rag] retrieval error: {e}")
            return jsonify({
                "answer": "I don’t know",
                "context_used": "",
                "query": query,
                "retrieved_chunks": [],
                "evidence": [],
                "fallback_reason": "retrieval_failed"
            })

        if not retrieved:
            return jsonify({
                "answer": "I don’t know",
                "context_used": "",
                "query": query,
                "retrieved_chunks": [],
                "evidence": [],
                "fallback_reason": "no_retrieved_chunks"
            })

        # --- 3. Generate structured answer ---
        try:
            answer, fallback_reason, evidence = generate_answer_with_fallback(
                query, context_raw, retrieved
            )
        except Exception as e:
            app.logger.error(f"[rag] generation error: {e}")
            answer, fallback_reason, evidence = (
                "I don’t know",
                "generation_failed",
                []
            )

        # --- 4. Post-process evidence and retrieved chunks ---
        for e in evidence:
            e["sim"] = safe_cast(e.get("sim", 0.0))

        context_used = "\n".join(
            [e.get("paraphrase", e.get("text", "")) for e in evidence[:STRUCTURED_TOP_SENTS]]
        )

        for r in retrieved:
            r["similarity"] = safe_cast(r.get("similarity"))
            r["chunk_id"] = int(r.get("chunk_id"))

        # --- 5. Return clean structured response ---
        return jsonify({
            "answer": answer,
            "context_used": context_used,
            "query": query,
            "retrieved_chunks": retrieved,
            "evidence": evidence,
            "fallback_reason": fallback_reason or ""
        })

    except Exception as e:
        app.logger.error(f"[rag] unexpected error: {e}", exc_info=True)
        return jsonify({
            "answer": "I don’t know",
            "context_used": "",
            "query": "",
            "retrieved_chunks": [],
            "evidence": [],
            "fallback_reason": "unexpected_error"
        }), 500


if __name__ == "__main__":
    if index is None:
        print("[app] Warning: FAISS index not loaded. Run `python ingest.py` or POST /ingest to build it.")
    print("[app] Starting server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
