
import logging

from backend.gemini_engine import GeminiEngine
from backend.groq_engine import GroqEngine


logger = logging.getLogger(__name__)


class HybridEngine:

    def __init__(self):

        self.gemini = None
        self.groq = None

        self.primary_provider = "gemini"
        self.fallback_provider = "groq"

        # Initialize Gemini if its API key exists.
        try:
            self.gemini = GeminiEngine(
                model="gemini-3.8-flash"
            )
        except RuntimeError as error:
            logger.warning(
                "Gemini initialization failed: %s", error
            )

        # Initialize Groq if its API key exists.
        try:
            self.groq = GroqEngine(
                model="openai/gpt-oss-20b"
            )
        except RuntimeError as error:
            logger.warning(
                "Groq initialization failed: %s", error
            )

        if self.gemini is None and self.groq is None:
            raise RuntimeError(
                "No AI provider is configured."
            )

    async def generate_response(
        self,
        prompt,
        max_new_tokens=512,
        temperature=0.4
    ):

        if self.gemini is not None:

            try:
                answer = await self.gemini.generate_response(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )

                logger.info("Response generated using Gemini.")
                return answer

            except RuntimeError as error:
                logger.warning(
                    "Gemini failed; trying Groq: %s", error
                )

        if self.groq is not None:

            try:
                answer = await self.groq.generate_response(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )

                logger.info("Response generated using Groq fallback.")
                return answer

            except RuntimeError as error:
                logger.error("Groq fallback failed: %s", error)

        raise RuntimeError(
            "LEO's AI providers are temporarily unavailable."
        )
