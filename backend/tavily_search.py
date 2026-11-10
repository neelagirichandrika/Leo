
import os
import httpx


class TavilySearchEngine:

    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com"

        if not self.api_key:
            raise RuntimeError(
                "TAVILY_API_KEY environment variable is missing."
            )

    async def search(self, query, max_results=5):

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers=headers
                )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException as error:
            raise RuntimeError(
                "Tavily search timed out."
            ) from error

        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"Tavily API error: HTTP "
                f"{error.response.status_code}: "
                f"{error.response.text[:300]}"
            ) from error

        except httpx.ConnectError as error:
            raise RuntimeError(
                "Cannot connect to Tavily."
            ) from error
