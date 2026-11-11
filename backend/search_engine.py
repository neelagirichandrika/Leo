
import os
import httpx


class TavilySearchEngine:

    def __init__(self):

        self.api_key = os.getenv("TAVILY_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "TAVILY_API_KEY environment variable is missing."
            )

        self.api_url = "https://api.tavily.com/search"

    async def search(self, query, max_results=5):

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": max_results,
            "include_answer": False
        }

        try:

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0)
            ) as client:

                response = await client.post(
                    self.api_url,
                    json=payload
                )

                response.raise_for_status()

                data = response.json()

            results = []

            for item in data.get("results", []):

                title = item.get("title", "").strip()
                url = item.get("url", "").strip()
                content = item.get("content", "").strip()

                if title and url and content:

                    results.append({
                        "title": title,
                        "url": url,
                        "content": content
                    })

            return results

        except httpx.TimeoutException as error:

            raise RuntimeError(
                "Tavily search request timed out."
            ) from error

        except httpx.ConnectError as error:

            raise RuntimeError(
                "Could not connect to Tavily."
            ) from error

        except httpx.HTTPStatusError as error:

            raise RuntimeError(
                f"Tavily API returned HTTP "
                f"{error.response.status_code}: "
                f"{error.response.text[:500]}"
            ) from error
