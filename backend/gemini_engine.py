
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

        self.retryable_statuses = {
            429, 500, 502, 503, 504
        }

        self.max_attempts = 2

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
                "maxOutputTokens": max_new_tokens,
                "temperature": temperature
            }
        }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        timeout = httpx.Timeout(
            connect=15.0,
            read=120.0,
            write=30.0,
            pool=15.0
        )

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:

            for attempt in range(1, self.max_attempts + 1):

                try:

                    print(
                        f"Sending request to Gemini "
                        f"(attempt {attempt}/{self.max_attempts})..."
                    )

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        json=payload
                    )

                    if response.status_code in self.retryable_statuses:

                        if attempt < self.max_attempts:

                            print(
                                f"Gemini returned HTTP "
                                f"{response.status_code}. Retrying..."
                            )

                            await asyncio.sleep(3)
                            continue

                        raise RuntimeError(
                            f"Gemini unavailable. HTTP "
                            f"{response.status_code}: "
                            f"{response.text[:500]}"
                        )

                    response.raise_for_status()

                    result = response.json()

                    candidates = result.get("candidates", [])

                    if not candidates:

                        raise RuntimeError(
                            "Gemini returned no response candidates. "
                            f"Details: {result.get('promptFeedback', {})}"
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

                    print("Gemini response received successfully.")

                    return answer

                except httpx.TimeoutException as error:

                    print(
                        f"Gemini request timed out "
                        f"(attempt {attempt})."
                    )

                    if attempt == self.max_attempts:

                        raise RuntimeError(
                            "Gemini timed out after "
                            f"{self.max_attempts} attempts. "
                            "The AI provider may be overloaded."
                        ) from error

                    await asyncio.sleep(3)

                except httpx.ConnectError as error:

                    if attempt == self.max_attempts:

                        raise RuntimeError(
                            "Cannot connect to Gemini API after "
                            f"{self.max_attempts} attempts."
                        ) from error

                    print("Gemini connection failed. Retrying...")
                    await asyncio.sleep(3)

                except httpx.HTTPStatusError as error:

                    status = error.response.status_code

                    raise RuntimeError(
                        f"Gemini API returned HTTP {status}: "
                        f"{error.response.text[:500]}"
                    ) from error
