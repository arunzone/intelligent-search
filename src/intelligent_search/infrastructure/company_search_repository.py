"""HTTP repository for Part 1 — the company search API."""

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from intelligent_search.domain.models import CompanySearchParams, CompanySearchResponse


class CompanySearchRepository:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    async def search(self, params: CompanySearchParams) -> CompanySearchResponse:
        query = params.model_dump(exclude_none=True, mode="json")
        logger.debug(f"Calling Part 1: GET /companies/search params={query}")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                f"{self._base_url}/companies/search", params=query
            )
            response.raise_for_status()
            data = CompanySearchResponse.model_validate(response.json())
            logger.debug(f"Part 1 returned total={data.total} results")
            return data
