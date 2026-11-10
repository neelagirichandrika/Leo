
import os
import httpx


class GroqEngine:

    def __init__(
        self,
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1"
    ):

        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is missing."
            )

        self.system_prompt = (
            "You are LEO, a helpful general-purpose AI assistant. "
            "Answer the user's exact question directly. "
            "Use simple, clear English. "
            "Be concise by default, but provide detailed answers "
            "when requested. "
            "For coding questions, provide correct code and explain it. "
            "Do not invent facts. If uncertain, say so."
        )

    async def generate_response(
        self,
        prompt,
        max_new_tokens=512,
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
            "max_completion_tokens": max(max_new_tokens, 2048),
            "temperature": temperature,
            "reasoning_effort": "low",
            "include_reasoning": False,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=15.0,
                    read=120.0,
                    write=30.0,
                    pool=15.0
                )
            ) as client:

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )

                response.raise_for_status()

                result = response.json()

                choices = result.get("choices", [])

                if not choices:
                    raise RuntimeError(
                        "Groq returned no response choices."
                    )

                choice = choices[0]
                message = choice.get("message", {})

                answer = message.get("content")

                if isinstance(answer, list):
                    answer = "\n".join(
                        item.get("text", "")
                        for item in answer
                        if isinstance(item, dict)
                    )

                if not isinstance(answer, str):
                    answer = ""

                answer = answer.strip()

                if not answer:
                    finish_reason = choice.get(
                        "finish_reason", "unknown"
                    )

                    raise RuntimeError(
                        "Groq returned no text. "
                        f"Finish reason: {finish_reason}. "
                        f"Model: {self.model}."
                    )

                return answer

        except httpx.TimeoutException as error:

            raise RuntimeError(
                "Groq API request timed out."
            ) from error

        except httpx.ConnectError as error:

            raise RuntimeError(
                "Cannot connect to Groq API."
            ) from error

        except httpx.HTTPStatusError as error:

            raise RuntimeError(
                f"Groq API returned HTTP "
                f"{error.response.status_code}: "
                f"{error.response.text[:500]}"
            ) from error
