import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize as sk_normalize
from rapidfuzz import process, fuzz
from symspellpy import SymSpell, Verbosity

# Config
BASE_DIR = Path(__file__).resolve().parent.parent  # points to GPTea/
KB_PATH = str(BASE_DIR / "data" / "domain.csv")
EMB_CACHE = str(BASE_DIR / "data" / "kb_embeddings.npy")
LOCAL_MODEL_PATH = str(BASE_DIR / "models" / "all-MiniLM-L6-v2")  # ← local folder, no download

# Spell checker setup
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)


def build_spell_dictionary(df: pd.DataFrame):
    """Build dictionary for typo correction from KB."""
    terms = set()

    for q in df["question"].dropna():
        terms.update(q.lower().split())

    if "tags" in df.columns:
        for t in df["tags"].dropna():
            if isinstance(t, str):
                terms.update(t.lower().split(";"))

    for word in terms:
        sym_spell.create_dictionary_entry(word.strip(), 100)


def correct_text(text: str) -> str:
    """Correct spelling mistakes in query."""
    if not isinstance(text, str):
        return ""

    words = text.lower().split()
    corrected = []

    for w in words:
        suggestions = sym_spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else w)

    return " ".join(corrected).strip()

# KB loading
def load_knowledge_base(path: str = KB_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge base not found at: {path}")

    df = pd.read_csv(path, dtype={"id": str})

    required = {"id", "question", "answer"}

    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"KB must contain columns: {required}. Found: {list(df.columns)}"
        )

    df["question_norm"] = df["question"].astype(str).str.lower().str.strip()
    df["id"] = df["id"].astype(str).str.strip()

    build_spell_dictionary(df)

    return df

# Embedding Model
def get_model_and_embeddings(
    df: pd.DataFrame,
    model_path: str = LOCAL_MODEL_PATH,
    cache_path: str = EMB_CACHE
):
    # SentenceTransformer accepts a local directory path directly
    model = SentenceTransformer(model_path)

    if os.path.exists(cache_path):

        try:
            kb_embeddings = np.load(cache_path)

            if kb_embeddings.shape[0] != len(df):
                raise ValueError("Embedding cache mismatch")

        except Exception:

            kb_embeddings = model.encode(
                df["question_norm"].tolist(),
                convert_to_numpy=True,
                show_progress_bar=True
            )

            np.save(cache_path, kb_embeddings)

    else:

        kb_embeddings = model.encode(
            df["question_norm"].tolist(),
            convert_to_numpy=True,
            show_progress_bar=True
        )

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        np.save(cache_path, kb_embeddings)

    kb_embeddings = sk_normalize(kb_embeddings, axis=1)

    return model, kb_embeddings

# Fuzzy fallback
def fuzzy_fallback(query_norm: str, choices: list, score_cutoff: int = 70):

    if not query_norm or not choices:
        return None

    match = process.extractOne(query_norm, choices, scorer=fuzz.token_sort_ratio)

    if not match:
        return None

    best_text, score, _ = match

    return best_text if score >= score_cutoff else None

# Main Retrieval Function
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

            fallback = fuzzy_fallback(
                query_corrected,
                df["question_norm"].tolist()
            )

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