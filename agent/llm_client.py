import time
from ollama import chat


# llama modelo y mide latencia
#qwen3:4b
#deepseek-r1:1.5b
#deepseek-r1:8b
def generate_response(prompt: str, model: str = "deepseek-r1:7b"):
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