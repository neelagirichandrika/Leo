
from datetime import datetime
from zoneinfo import ZoneInfo


class LEOKnowledgeEngine:

    def __init__(self, ai_engine, search_engine):
        self.ai_engine = ai_engine
        self.search_engine = search_engine

    def get_current_datetime(self):
        now = datetime.now(ZoneInfo("Asia/Kolkata"))

        return now.strftime(
            "%A, %d %B %Y, %I:%M %p IST"
        )

    def needs_web_search(self, question):

        keywords = [
            "latest", "today", "current", "currently",
            "recent", "news", "now", "this week",
            "this month", "releasing", "release date",
            "price", "weather", "2026", "2027",
            "upcoming", "breaking", "yesterday",
            "tomorrow", "this year"
        ]

        question = question.lower()

        return any(
            keyword in question
            for keyword in keywords
        )

    async def generate_response(
        self,
        question,
        max_new_tokens=512,
        temperature=0.4
    ):

        if not self.needs_web_search(question):

            return await self.ai_engine.generate_response(
                prompt=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

        current_datetime = self.get_current_datetime()

        results = await self.search_engine.search(
            query=question,
            max_results=5
        )

        sources = results.get("results", [])

        if not sources:

            return await self.ai_engine.generate_response(
                prompt=(
                    f"Current date and time: {current_datetime}\n"
                    f"User question: {question}\n"
                    "No web search results were found. "
                    "Do not pretend that live information was verified."
                ),
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

        research_context = ""

        for index, source in enumerate(sources, 1):

            research_context += (
                f"\nSource {index}:\n"
                f"Title: {source.get('title', '')}\n"
                f"URL: {source.get('url', '')}\n"
                f"Information: {source.get('content', '')}\n"
            )

        prompt = f"""
Current date and time:
{current_datetime}

User question:
{question}

Web research results:
{research_context}

Instructions:

1. Answer the user's exact question.
2. Use the supplied search results for current information.
3. Distinguish past, ongoing and future events using the current date.
4. Never describe a past event as upcoming.
5. Do not invent facts, dates, names or statistics.
6. If sources disagree, explain the disagreement.
7. Cite relevant sources using Markdown links.
8. Clearly identify unconfirmed information.
9. Do not claim that information is verified if the sources
   do not support it.
10. Use simple, clear English.
11. Organize the answer with headings or tables when useful.
"""

        return await self.ai_engine.generate_response(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
