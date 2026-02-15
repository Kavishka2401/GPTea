"""
rag_retrieval.py — with SymSpell spell correction

Adds automatic typo correction using SymSpell:
- Learns from your KB questions and tags
- Corrects queries like "humidte" → "humidity"
- Replaces hardcoded COMMON_REPLACEMENTS

Dependencies:
  pip install symspellpy sentence-transformers scikit-learn rapidfuzz numpy pandas
"""

import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize as sk_normalize
from rapidfuzz import process, fuzz
from symspellpy import SymSpell, Verbosity

# Config
KB_PATH = "data/domain.csv"
EMB_CACHE = "data/kb_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------------
# Spell checker setup
# -------------------------
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

def build_spell_dictionary(df: pd.DataFrame):
    terms = set()
    for q in df["question"].dropna():
        terms.update(q.lower().split())
    for t in df.get("tags", []):
        if isinstance(t, str):
            terms.update(t.lower().split(";"))
    for word in terms:
        sym_spell.create_dictionary_entry(word.strip(), 100)

def correct_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    words = text.lower().split()
    corrected = []
    for w in words:
        suggestions = sym_spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else w)
    return " ".join(corrected).strip()

# -------------------------
# KB loading and embedding
# -------------------------
def load_knowledge_base(path: str = KB_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge base not found at: {path}")
    df = pd.read_csv(path, dtype={"id": str})
    required = {"id", "question", "answer"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"KB must contain columns: {required}. Found: {list(df.columns)}")
    df["question_norm"] = df["question"].astype(str).str.lower().str.strip()
    df["id"] = df["id"].astype(str).str.strip()
    build_spell_dictionary(df)
    return df

def get_model_and_embeddings(df: pd.DataFrame, model_name: str = MODEL_NAME, cache_path: str = EMB_CACHE):
    model = SentenceTransformer(model_name)
    if os.path.exists(cache_path):
        try:
            kb_embeddings = np.load(cache_path)
            if kb_embeddings.shape[0] != len(df):
                raise ValueError("Embedding cache size mismatch")
        except Exception:
            kb_embeddings = model.encode(df["question_norm"].tolist(), convert_to_numpy=True, show_progress_bar=True)
            np.save(cache_path, kb_embeddings)
    else:
        kb_embeddings = model.encode(df["question_norm"].tolist(), convert_to_numpy=True, show_progress_bar=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, kb_embeddings)
    kb_embeddings = sk_normalize(kb_embeddings, axis=1)
    return model, kb_embeddings

# -------------------------
# Retrieval
# -------------------------
def fuzzy_fallback(query_norm: str, choices: list, score_cutoff: int = 70):
    if not query_norm or not choices:
        return None
    match = process.extractOne(query_norm, choices, scorer=fuzz.token_sort_ratio)
    if not match:
        return None
    best_text, score, _ = match
    return best_text if score >= score_cutoff else None

def retrieve_docs(query: str, k: int = 3, min_score: float = 0.45, path: str = KB_PATH):
    query_corrected = correct_text(query)
    if len(query_corrected) < 2:
        return {"error": "Query too short or invalid."}

    df = load_knowledge_base(path)
    model, kb_embeddings = get_model_and_embeddings(df)

    q_emb = model.encode([query_corrected], convert_to_numpy=True)
    q_emb = sk_normalize(q_emb, axis=1)

    sims = np.dot(kb_embeddings, q_emb.T).squeeze()
    if np.isscalar(sims):
        sims = np.array([sims])

    k = min(k, len(sims))
    top_idx = sims.argsort()[-k:][::-1]

    results = []
    for idx in top_idx:
        score = float(sims[idx])
        row = df.iloc[idx]
        if score < min_score:
            fallback = fuzzy_fallback(query_corrected, df["question_norm"].tolist())
            if fallback:
                fb_idx_list = df.index[df["question_norm"] == fallback].tolist()
                if fb_idx_list:
                    row = df.loc[fb_idx_list[0]]
                    score = 0.5
                else:
                    continue
            else:
                continue
        results.append({
            "id": row["id"],
            "intent": row.get("intent", ""),
            "question": row["question"],
            "answer": row["answer"],
            "tags": row.get("tags", ""),
            "score": round(score, 3)
        })

    if not results:
        return {"message": "No confident matches. Try rephrasing or provide more details."}
    return results

def print_results(results):
    if isinstance(results, dict):
        if results.get("error"):
            print("Error:", results["error"])
        else:
            print(results.get("message", json.dumps(results, indent=2)))
        return
    for r in results:
        print(f"Score: {r['score']}  ID: {r['id']}")
        print(f"Q: {r['question']}")
        print(f"A: {r['answer']}")
        if r.get("tags"):
            print(f"Tags: {r['tags']}")
        print("-" * 60)

# -------------------------
# CLI test
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG retrieval test with SymSpell")
    parser.add_argument("--query", type=str, default="how to manage humidte in tea", help="Query text")
    parser.add_argument("--k", type=int, default=3, help="Top-k results")
    parser.add_argument("--kb", type=str, default=KB_PATH, help="Path to KB CSV")
    args = parser.parse_args()

    try:
        docs = retrieve_docs(args.query, k=args.k, path=args.kb)
        print_results(docs)
    except Exception as e:
        print("Error during retrieval:", repr(e))
