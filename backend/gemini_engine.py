
import os
import asyncio
import httpx


class GeminiEngine:

    def __init__(self, model="gemini-3.8-flash"):

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
                "parts": [
                    {"text": self.system_prompt}
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_new_tokens
            }
        }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        retryable_statuses = {429, 500, 502, 503, 504}
        delays = [2, 4, 8]

        try:

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    90.0,
                    connect=10.0
                )
            ) as client:

                for attempt in range(4):

                    try:

                        response = await client.post(
                            self.api_url,
                            headers=headers,
                            json=payload
                        )

                        response.raise_for_status()

                        result = response.json()

                        candidates = result.get(
                            "candidates", []
                        )

                        if not candidates:
                            raise RuntimeError(
                                "Gemini returned no response candidates."
                            )

                        parts = candidates[0].get(
                            "content", {}
                        ).get("parts", [])

                        answer = "".join(
                            part.get("text", "")
                            for part in parts
                        ).strip()

                        if not answer:
                            raise RuntimeError(
                                "Gemini returned an empty response."
                            )

                        return answer

                    except httpx.HTTPStatusError as error:

                        status = error.response.status_code

                        if status not in retryable_statuses:
                            raise RuntimeError(
                                f"Gemini API returned HTTP {status}: "
                                f"{error.response.text[:500]}"
                            ) from error

                        if attempt == 3:
                            raise RuntimeError(
                                f"Gemini is unavailable after 4 attempts. "
                                f"HTTP {status}: "
                                f"{error.response.text[:300]}"
                            ) from error

                        delay = delays[attempt]

                        print(
                            f"Gemini returned HTTP {status}. "
                            f"Retrying in {delay} seconds..."
                        )

                        await asyncio.sleep(delay)

        except httpx.ConnectError as error:

            raise RuntimeError(
                "Cannot connect to Gemini API."
            ) from error

        except httpx.TimeoutException as error:

            raise RuntimeError(
                "Gemini API request timed out."
            ) from error
