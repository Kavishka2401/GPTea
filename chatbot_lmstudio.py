import requests
from rag_retrieval import retrieve_docs

# LM Studio local server endpoint
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"


def ask_llm_with_rag(query: str):
    """
    Uses RAG to retrieve relevant tea knowledge chunks,
    then sends context + question to the LLM for final answer.
    """

    # STEP 1 — Retrieve relevant knowledge from your CSV
    docs = retrieve_docs(query, k=3)

    # If no good match, return fallback
    if docs == []:
        return (
            "Sorry, I don’t have enough information in my tea knowledge base "
            "to answer that question."
        )

    # Combine text chunks into a single context block
    context = "\n\n".join([d["text_chunk"] for d in docs])

    # STEP 2 — Build the LLM prompt
    prompt = f"""
You are a Sri Lankan tea farming assistant.
Use ONLY the information in the context below.
If the context does not contain the answer, say:
"I don't know based on the provided data."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    # STEP 3 — Send to LM Studio local model
    response = requests.post(
        LMSTUDIO_URL,
        json={
            "model": "local-model",      # LM Studio automatically maps this
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
    )

    # Extract LLM text response
    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Tea Assistant (RAG + LLM) Ready!")
    print("Ask anything about tea farming. Type 'exit' to quit.\n")

    while True:
        user_q = input("Farmer: ")

        if user_q.lower() in ("exit", "quit"):
            break

        answer = ask_llm_with_rag(user_q)
        print("\nAssistant:", answer, "\n")
