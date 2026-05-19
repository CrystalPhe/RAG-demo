import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_APP_TITLE,
)


def ask_openrouter(question: str, retrieved) -> str:
    if not OPENROUTER_API_KEY:
        return "Set OPENROUTER_API_KEY to generate an AI answer from the retrieved Qdrant context."

    context = retrieved
    if not context:
        return "I could not find a strong match in the uploaded PDF, so there is not enough context to reason over."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded PDF question-answering assistant. "
                "Answer only from the context provided from Qdrant. "
                "Do not use outside knowledge. "
                "If the context is insufficient, say exactly what is missing and that you cannot confirm the answer. "
                "Be concise and cite page/chunk numbers inline when you use them. "
                "Reply in the same language as the user question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved context from Qdrant:\n{context}"
            ),
        },
    ]

    payload = json.dumps(
        {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "temperature": 0.2,
        }
    ).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE

    request = Request(OPENROUTER_BASE_URL, data=payload, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=60) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"OpenRouter request failed: {exc.code} {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

    data = json.loads(response_text)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter returned no choices.")

    message = choices[0].get("message") or {}
    content = str(message.get("content", "")).strip()
    if not content:
        raise RuntimeError("OpenRouter returned an empty response.")
    return content
