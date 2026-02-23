import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain_core.evaluation import load_evaluator
from chatbot_openai import chain  # Import your new LangChain chain

# 1. SETUP
load_dotenv()
judge_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))


def validate_bot():
    # Load your test cases
    df = pd.read_csv('dataset.csv')
    print(f"Starting Validation on {len(df)} tea farming samples...")

    results = []
    correct_count = 0

    # Initialize the LangChain Correctness Evaluator
    # This uses a "Judge" to compare the bot's answer to the Ground Truth
    evaluator = load_evaluator("labeled_criteria", criteria="correctness", llm=None)
    # Note: For a custom judge prompt like yours, we use your existing judge logic:

    for idx, row in df.iterrows():
        question = row['question']
        ground_truth = row['answer']

        # Step A: Get the Chatbot's Answer using your LangChain Chain
        try:
            generated_answer = chain.invoke(question)
        except Exception as e:
            generated_answer = f"ERROR: {e}"

        # Step B: LLM Judge Logic
        verdict_prompt = f"""
        Judge the following farming advice.
        QUESTION: {question}
        GROUND TRUTH: {ground_truth}
        BOT ANSWER: {generated_answer}

        Is the BOT ANSWER factually consistent with the GROUND TRUTH? 
        Respond ONLY with 'CORRECT' or 'INCORRECT'.
        """

        try:
            judge_res = judge_llm.chat.completions.create(
                messages=[{"role": "user", "content": verdict_prompt}],
                model="llama-3.1-8b-instant"
            )
            verdict = judge_res.choices[0].message.content.strip().upper()
        except:
            verdict = "ERROR"

        if "CORRECT" in verdict:
            correct_count += 1

        results.append({
            "Question": question,
            "Ground Truth": ground_truth,
            "Bot Answer": generated_answer,
            "Verdict": verdict
        })
        print(f"[{idx + 1}/{len(df)}] Verdict: {verdict}")

    # 3. FINAL METRICS
    accuracy = (correct_count / len(df)) * 100
    print(f"\n" + "=" * 30)
    print(f"FINAL ACCURACY: {accuracy:.2f}%")
    print("=" * 30)

    # 4. SAVE REPORT
    report_df = pd.DataFrame(results)
    report_df.to_csv("accuracy_report.csv", index=False)
    print("Report saved to 'accuracy_report.csv'")
    #checkkkk


if __name__ == "__main__":
    validate_bot()