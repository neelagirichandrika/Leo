
import os
import httpx


class GeminiEngine:
    def __init__(self, model="gemini-2.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = model

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is missing."
            )

        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent"
        )

        self.system_prompt = (
            "You are LEO, a helpful general-purpose AI assistant. "
            "Answer the user's exact question directly. "
            "Use simple, clear English. "
            "Be concise by default, but provide detailed answers "
            "when requested. "
            "For coding questions, provide correct code and explain it. "
            "Do not invent facts. If you are uncertain, say so."
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
            "systemInstruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_new_tokens
            }
        }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=10.0)
            ) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )

                response.raise_for_status()
                result = response.json()

                candidates = result.get("candidates", [])
                if not candidates:
                    raise RuntimeError(
                        "Gemini returned no response candidates."
                    )

                parts = candidates[0].get(
                    "content", {}
                ).get("parts", [])

                answer = "".join(
                    part.get("text", "") for part in parts
                ).strip()

                if not answer:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return answer

        except httpx.ConnectError as error:
            raise RuntimeError(
                "Cannot connect to Gemini API."
            ) from error

        except httpx.TimeoutException as error:
            raise RuntimeError(
                "Gemini API request timed out."
            ) from error

        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"Gemini API returned HTTP "
                f"{error.response.status_code}: "
                f"{error.response.text[:500]}"
            ) from error
