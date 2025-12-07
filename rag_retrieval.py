import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load XLSX instead of CSV
df = pd.read_excel("qa_chunks.xlsx")

texts = df["text_chunk"].tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
 # shape: (n_chunks, n_features)

# ---- 2. Retrieval function ----
def retrieve_docs(query: str, k: int = 3, threshold: float = 0.1):
    """
    Retrieve top-k relevant knowledge chunks for RAG.
    If relevance is too low, return empty list.
    """
    # Convert user query to TF-IDF vector
    q_vec = vectorizer.transform([query])

    # Cosine similarity between query and all chunks
    similarities = linear_kernel(q_vec, tfidf_matrix).flatten()

    # Get top-k best matches
    top_idx = similarities.argsort()[::-1][:k]
    top_scores = similarities[top_idx]

    # Fallback: no good match
    if max(top_scores) < threshold:
        return []

    # Return results as a list of dicts
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


# ---- 3. Manual testing ----
if __name__ == "__main__":
    while True:
        q = input("\nAsk me something: ")
        if q.lower() == "quit":
            break

        results = retrieve_docs(q)
        print("\nTop Results:")
        for r in results:
            print(f"\n[{r['id']}], score={r['score']:.3f}")
            print(r["text_chunk"])
