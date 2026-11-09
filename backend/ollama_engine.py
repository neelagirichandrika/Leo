
import httpx


# ==========================================
# LEO - OLLAMA INTELLIGENCE ENGINE
# ==========================================

class OllamaEngine:

    def __init__(
        self,
        model="qwen3.5:4b",
        base_url="http://localhost:11434"
    ):

        self.model = model
        self.base_url = base_url.rstrip("/")

        self.system_prompt = (
            "You are LEO, a helpful AI assistant. "
            "Answer the exact question directly. "
            "Use simple, clear English. "
            "Be concise by default: 2 to 5 sentences "
            "for ordinary questions. "
            "Do not add unrelated information, unnecessary "
            "introductions, or repeated conclusions. "
            "For coding questions, provide correct code "
            "and a brief explanation. "
            "Give detailed answers when requested. "
            "If you do not know something, say so."
        )

    async def generate_response(
        self,
        prompt,
        max_new_tokens=150,
        temperature=0.4
    ):

        if not prompt or not prompt.strip():
            raise ValueError("Message cannot be empty.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "think": False,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens
            }
        }

        try:

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(180.0, connect=10.0)
            ) as client:

                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )

                response.raise_for_status()

                result = response.json()

                answer = result.get(
                    "message", {}
                ).get("content", "")

                if not answer or not answer.strip():
                    raise RuntimeError(
                        "Ollama returned an empty response."
                    )

                return answer.strip()

        except httpx.ConnectError as error:

            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Make sure Ollama is running."
            ) from error

        except httpx.TimeoutException as error:

            raise RuntimeError(
                "Ollama response timed out."
            ) from error

        except httpx.HTTPStatusError as error:

            raise RuntimeError(
                f"Ollama API returned HTTP "
                f"{error.response.status_code}: "
                f"{error.response.text[:500]}"
            ) from error
