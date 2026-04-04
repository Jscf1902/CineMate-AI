from ollama import chat


# llamada simple al modelo
def generate_response(prompt: str, model: str = "qwen3:1.7b") -> str:
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]