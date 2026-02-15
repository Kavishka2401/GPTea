import os
import re
from google import genai  # UPDATED: Modern GenAI import
from groq import Groq
from dotenv import load_dotenv
from rag_retrieval import retrieve_docs
from recommendation_engine import RecommendationEngine

# 1. Load environment variables
load_dotenv()

# 2. Initialize API Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# UPDATED: New Gemini Client setup for 2026
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize recommendation engine
engine = RecommendationEngine()

# 3. Utility Functions
COMMON_QUERY_REPLACEMENTS = {
    "netrogen": "nitrogen", "contatent": "content",
    "leves": "leaves", "yllow": "yellow", "ph": "ph"
}


def normalize_query(q: str) -> str:
    if not isinstance(q, str): return ""
    s_lower = q.strip().lower()
    for bad, good in COMMON_QUERY_REPLACEMENTS.items():
        s_lower = s_lower.replace(bad, good)
    return " ".join(s_lower.split())


def call_gemini_fallback(query: str):
    """
    UPDATED: Using Gemini 2.0 Flash to avoid 404 errors.
    """
    prompt = f"You are a professional tea farming assistant. Answer this based on general knowledge: {query}"
    try:
        # We use 'gemini-2.0-flash' which is the 2026 standard
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return f"{response.text}\n\n(Note: This info is from Gemini 2.0 general knowledge.)"

    except Exception as e:
        # If 2.0 fails, we try the absolute latest "Flash Lite" as a last resort
        try:
            print("Trying 2.0 Flash-Lite fallback...")
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt
            )
            return response.text
        except:
            return f"I'm sorry, I couldn't find an answer in our data or via fallback. Error: {e}"

# 4. Main RAG Logic
def ask_llm_with_rag(query: str):
    query_norm = normalize_query(query)
    docs = retrieve_docs(query_norm, k=3)

    context = ""
    # Check if retrieval found anything in your CSV/Knowledge Base
    if isinstance(docs, dict):
        if docs.get("message") or docs.get("error"):
            return call_gemini_fallback(query)
    elif isinstance(docs, list):
        if not docs:
            return call_gemini_fallback(query)
        context = "\n\n".join([d.get("answer", "") for d in docs])
    else:
        return call_gemini_fallback(query)

    # Build the strict prompt for Groq
    prompt = f"""
    You are a Sri Lankan tea farming assistant.
    Use ONLY the information in the context below.
    If the context does not contain the answer, literally say "TRIGGER_FALLBACK".

    CONTEXT:
    {context}

    QUESTION:
    {query}
    """

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Low temperature for high accuracy
        )
        msg = response.choices[0].message.content

        # If Groq can't find it in your data, trigger Gemini
        if "TRIGGER_FALLBACK" in msg or "I don't know" in msg:
            return call_gemini_fallback(query)

        return msg
    except Exception:
        return call_gemini_fallback(query)


# 5. Numerical Helpers
def extract_numeric_value(query: str):
    match = re.search(r"(\d+(\.\d+)?)", query)
    return float(match.group(1)) if match else None


# 6. Interaction Loop
if __name__ == "__main__":
    print("GPTea Hybrid Assistant (Local RAG + Gemini Fallback) Ready!")
    while True:
        user_q = input("Farmer: ")
        if user_q.lower() in ("exit", "quit"): break

        user_q_norm = normalize_query(user_q)
        answer = ask_llm_with_rag(user_q)

        # Handle Recommendations
        recs = []
        if "sodium" in user_q_norm:
            val = extract_numeric_value(user_q)
            if val is not None: recs.extend(engine.recommend_from_soil({"sodium": val}))
        if "ph" in user_q_norm:
            val = extract_numeric_value(user_q)
            if val is not None: recs.extend(engine.recommend_from_soil({"pH": val}))
        if "brown blight" in user_q.lower():
            recs.extend(engine.recommend_from_pest("Brown Blight"))

        final_output = f"{answer}\nAdditionally: {'; '.join(recs)}" if recs else answer
        print(f"\nAssistant: {final_output}\n")