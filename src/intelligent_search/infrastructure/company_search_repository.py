"""HTTP repository for Part 1 — the company search API."""

from typing import Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class CompanySearchRepository:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    async def search(
        self,
        name: Optional[str] = None,
        industry: Optional[str] = None,
        location: Optional[str] = None,
        founded_year_min: Optional[int] = None,
        founded_year_max: Optional[int] = None,
        size_range: Optional[str] = None,
        page: int = 1,
        size: int = 10,
    ) -> dict:
        params: dict = {"page": page, "size": size}
        if name:
            params["name"] = name
        if industry:
            params["industry"] = industry
        if location:
            params["location"] = location
        if founded_year_min:
            params["founded_year_min"] = founded_year_min
        if founded_year_max:
            params["founded_year_max"] = founded_year_max
        if size_range:
            params["size_range"] = size_range

        logger.debug(f"Calling Part 1: GET /companies/search params={params}")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                f"{self._base_url}/companies/search", params=params
            )
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Part 1 returned total={data.get('total')} results")
            return data
