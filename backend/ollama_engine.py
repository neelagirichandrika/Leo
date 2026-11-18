
import time
import httpx


# ==========================================
# LEO - OLLAMA INTELLIGENCE ENGINE
# ==========================================

class OllamaEngine:

    def __init__(
        self,
        model="llama3.2:3b",
        base_url="http://127.0.0.1:11434"
    ):

        self.model = model
        self.base_url = base_url.rstrip("/")

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                180.0,
                connect=10.0
            )
        )

        self.system_prompt = (
            "You are LEO, a helpful AI assistant. "
            "Answer the exact question directly. "
            "Use simple, clear English. "
            "Be concise by default. "
            "Do not add unrelated information or repeat conclusions. "
            "For coding questions, provide correct code "
            "and a brief explanation. "
            "Give detailed answers when requested. "
            "Use conversation history to understand follow-up questions. "
            "Do not treat every message as a new conversation. "
            "If you do not know something, say so. "
            "When web research is provided, use its evidence "
            "and cite sources using the supplied reference numbers."
        )

    # ==========================================
    # GENERATE RESPONSE
    # ==========================================

    async def generate_response(
        self,
        prompt,
        max_new_tokens=150,
        temperature=0.4,
        history=None
    ):

        if not prompt or not prompt.strip():

            raise ValueError(
                "Message cannot be empty."
            )

        # BUILD MESSAGE HISTORY

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        if history:

            for item in history:

                role = item.get("role")
                content = item.get("content", "")

                if (
                    role in ("user", "assistant")
                    and isinstance(content, str)
                    and content.strip()
                ):

                    messages.append({
                        "role": role,
                        "content": content
                    })

        # ADD CURRENT USER MESSAGE

        messages.append({
            "role": "user",
            "content": prompt
        })

        # OLLAMA REQUEST

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens
            }
        }

        try:

            print("\n===== OLLAMA GENERATION STARTED =====")

            print(f"Model: {self.model}")
            print(f"Messages sent: {len(messages)}")
            print(f"Maximum output tokens: {max_new_tokens}")

            start_time = time.perf_counter()

            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )

            response.raise_for_status()

            result = response.json()

            total_time = time.perf_counter() - start_time

            # EXTRACT ANSWER

            answer = result.get(
                "message", {}
            ).get("content", "")

            if not answer or not answer.strip():

                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            # OLLAMA PERFORMANCE METRICS

            prompt_eval_count = result.get(
                "prompt_eval_count", 0
            )

            prompt_eval_duration = result.get(
                "prompt_eval_duration", 0
            )

            eval_count = result.get(
                "eval_count", 0
            )

            eval_duration = result.get(
                "eval_duration", 0
            )

            load_duration = result.get(
                "load_duration", 0
            )

            # NANOSECONDS TO SECONDS

            prompt_eval_seconds = (
                prompt_eval_duration / 1_000_000_000
            )

            generation_seconds = (
                eval_duration / 1_000_000_000
            )

            load_seconds = (
                load_duration / 1_000_000_000
            )

            # TOKENS PER SECOND

            tokens_per_second = 0

            if eval_duration > 0:

                tokens_per_second = (
                    eval_count /
                    (eval_duration / 1_000_000_000)
                )

            # PRINT METRICS

            print("\n===== OLLAMA PERFORMANCE =====")

            print(
                f"Total HTTP request time: "
                f"{total_time:.2f} seconds"
            )

            print(
                f"Model loading time: "
                f"{load_seconds:.2f} seconds"
            )

            print(
                f"Prompt processing time: "
                f"{prompt_eval_seconds:.2f} seconds"
            )

            print(
                f"Prompt tokens: {prompt_eval_count}"
            )

            print(
                f"AI generation time: "
                f"{generation_seconds:.2f} seconds"
            )

            print(
                f"Generated tokens: {eval_count}"
            )

            print(
                f"Generation speed: "
                f"{tokens_per_second:.2f} tokens/second"
            )

            print("===== OLLAMA GENERATION COMPLETED =====\n")

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

    # ==========================================
    # CLOSE HTTP CLIENT
    # ==========================================

    async def close(self):

        if not self.client.is_closed:

            await self.client.aclose()
