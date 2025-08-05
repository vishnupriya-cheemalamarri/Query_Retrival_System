from typing import List
import requests

def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n".join(f"- {c}" for c in context_chunks)
    return f"""Context:\n{context}\n\nQuestion: {question}\nAnswer:"""

def generate_answer(question: str, context_chunks: List[str]) -> str:
    prompt = build_prompt(question, context_chunks)

    # Call free hosted model
    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-small",
        headers={"Accept": "application/json"},
        json={"inputs": prompt}
    )

    try:
        return response.json()[0]["generated_text"]
    except Exception:
        return "Error or no response from model"
