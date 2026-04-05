import time
from ollama import chat


# llama modelo y mide latencia
def generate_response(prompt: str, model: str = "qwen3:1.7b"):
    start = time.time()

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    end = time.time()

    latency_ms = int((end - start) * 1000)

    return {
        "content": response["message"]["content"],
        "latency_ms": latency_ms
    }