import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# ----------------------------------------------------
# 1. LOAD KNOWLEDGE BASE (CSV)
# ----------------------------------------------------

print("Loading Q&A chunks...")

# Load the CSV version
df = pd.read_csv("qa_chunks.csv")

# Extract only the text column
texts = df["text_chunk"].tolist()


# ----------------------------------------------------
# 2. BUILD TF-IDF MATRIX
# ----------------------------------------------------

print("Fitting TF-IDF vectorizer...")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)   # shape: (n_chunks, n_features)


# ----------------------------------------------------
# 3. RETRIEVAL FUNCTION
# ----------------------------------------------------

def retrieve_docs(query: str, k: int = 3, threshold: float = 0.1):
    """
    Retrieve top-k relevant knowledge chunks for RAG.
    If similarity is too low, return empty list (fallback).
    """

    # Convert user query into TF-IDF vector
    q_vec = vectorizer.transform([query])

    # Compute cosine similarity with all chunks
    similarities = linear_kernel(q_vec, tfidf_matrix).flatten()

    # Sort chunks by similarity (descending)
    top_idx = similarities.argsort()[::-1][:k]
    top_scores = similarities[top_idx]

    # Fallback if no good match
    if max(top_scores) < threshold:
        return []

    # Build result list
    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "id": row["id"],
            "intent": row["intent"],
            "score": float(similarities[idx]),
            "text_chunk": row["text_chunk"]
        })

    return results


# ----------------------------------------------------
# 4. MANUAL TESTING
# ----------------------------------------------------

if __name__ == "__main__":
    print("\nRAG Ready! Ask me something.")
    while True:
        query = input("\nYour question (or 'quit'): ")

        if query.lower() in ("quit", "exit"):
            break

        docs = retrieve_docs(query)

        if docs == []:
            print("\nNo relevant context found. (Fallback)")
        else:
            print("\nTop Retrieved Chunks:")
            for d in docs:
                print(f"\n[ID {d['id']}]  score={d['score']:.3f}")
                print(d["text_chunk"])
