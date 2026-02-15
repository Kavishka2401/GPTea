import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from chatbot_openai import ask_llm_with_rag
from rag_retrieval import retrieve_docs

# 1. SETUP
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_llm_judge_score(question, ground_truth, generated_answer):
    """Asks the LLM to act as a judge and score the answer."""
    prompt = f"""
    You are an expert judge evaluating a farming assistant.

    Question: {question}
    Correct Ground Truth: {ground_truth}
    Chatbot Answer: {generated_answer}

    Task: Is the Chatbot Answer factually consistent with the Ground Truth? 
    Respond with ONLY one word: 'CORRECT' or 'INCORRECT'.
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",  # Using a faster/cheaper model for judging
        )
        return response.choices[0].message.content.strip().upper()
    except:
        return "ERROR"


# 2. RUN SAMPLES
df = pd.read_csv('dataset.csv')
print(f"Starting Simplified Validation on {len(df)} samples...")

results = []
correct_count = 0

for idx, row in df.iterrows():
    q = row['question']
    gt = row['answer']

    # Get Chatbot Response
    ans = ask_llm_with_rag(q)

    # Judge the response
    verdict = get_llm_judge_score(q, gt, ans)

    if "CORRECT" in verdict:
        correct_count += 1

    results.append({
        "Question": q,
        "Ground Truth": gt,
        "Chatbot Answer": ans,
        "Judge Verdict": verdict
    })
    print(f"[{idx + 1}/{len(df)}] Verdict: {verdict}")

# 3. SHOW FINAL ACCURACY
accuracy = (correct_count / len(df)) * 100
print(f"\n{'=' * 30}")
print(f"FINAL ACCURACY: {accuracy:.2f}%")
print(f"{'=' * 30}")

# 4. SAVE TO CSV
pd.DataFrame(results).to_csv("simple_accuracy_report.csv", index=False)
print("Detailed report saved to 'simple_accuracy_report.csv'")