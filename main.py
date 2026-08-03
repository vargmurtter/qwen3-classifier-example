from typing import Literal

from ollama import Client
from pydantic import BaseModel

HOST = "http://192.168.1.18:11434"

CATEGORIES = [
    "payment",
    "technical issue",
    "documents",
]


class Result(BaseModel):
    category: Literal[
        "payment",
        "technical issue",
        "documents",
    ]
    summary: str
    confidence: float


def classify_email(text: str) -> dict:
    client = Client(host=HOST)

    response = client.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Классифицируй обращение по одной из категорий: "
                    f"{', '.join(CATEGORIES)}. "
                    "Кратко перескажи суть. "
                    "Укажи уверенность от 0 до 1. "
                ),
            },
            {"role": "user", "content": text},
        ],
        format=Result.model_json_schema(),
        options={"temperature": 0},
    )

    if response.message.content is None:
        return {}

    result = Result.model_validate_json(response.message.content)

    category = result.category
    if result.confidence < 0.7:
        category = "unsorted"

    return {
        "category": category,
        "summary": result.summary,
        "confidence": result.confidence,
    }


email = "в отчётной документации ошибка - неправильно написали моё имя"
print(classify_email(email))
