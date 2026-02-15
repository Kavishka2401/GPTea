import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup & API Keys
load_dotenv()

# 2. Define your LLMs (The "Smart" Parts)
# We define Groq as primary and Gemini as the fallback
primary_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)

# This is the "Safety Net": If Groq fails, it uses Gemini automatically
smart_llm = primary_llm.with_fallbacks([fallback_llm])

# 3. Define the Prompt (The "Instructions")
template = """
You are a Sri Lankan tea farming assistant.
Use the following context to answer the question. 
If the context is empty or doesn't have the answer, use your general knowledge but mention it's general advice.

CONTEXT:
{context}

QUESTION: 
{question}

ANSWER:
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. The Recommendation Logic (Integrated as a Tool)
from recommendation_engine import RecommendationEngine

engine = RecommendationEngine()


def get_recommendations(query, answer):
    """Checks for soil/pest keywords and appends recommendations."""
    recs = []
    query_lower = query.lower()
    # Simple keyword detection (can be improved with LangChain tools later)
    if "ph" in query_lower: recs.extend(engine.recommend_from_soil({"pH": 6.0}))  # Example
    if "brown blight" in query_lower: recs.extend(engine.recommend_from_pest("Brown Blight"))

    if recs:
        return f"{answer}\n\nAdditionally: {'; '.join(recs)}"
    return answer


# 5. The Retrieval Wrapper
# (Assumes your rag_retrieval.py returns a string of context)
from rag_retrieval import retrieve_docs


def get_context(query):
    docs = retrieve_docs(query, k=3)
    if isinstance(docs, list):
        return "\n".join([d.get("answer", "") for d in docs])
    return ""


# 6. Build the Chain
# This pipeline: Takes input -> Gets Context -> Formats Prompt -> Calls AI -> Parses Text
chain = (
        {"context": get_context, "question": RunnablePassthrough()}
        | prompt
        | smart_llm
        | StrOutputParser()
)

# 7. Execution Loop
if __name__ == "__main__":
    print("LangChain Tea Assistant Ready!")
    while True:
        user_input = input("Farmer: ")
        if user_input.lower() in ["exit", "quit"]: break

        # Run the chain
        raw_answer = chain.invoke(user_input)

        # Add recommendations
        final_answer = get_recommendations(user_input, raw_answer)
        print(f"\nAssistant: {final_answer}\n")