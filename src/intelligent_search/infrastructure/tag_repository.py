"""HTTP repository for the Part 1 tag endpoints."""

import httpx
from loguru import logger
from pydantic import BaseModel

from intelligent_search.domain.models import (
    CompanyTagsResponse,
    Tag,
    TagCreate,
    TagSummary,
    TagType,
)


class _RemoveTagParams(BaseModel):
    tag_type: TagType
    user_id: str | None = None


class _ListTagsParams(BaseModel):
    user_id: str | None = None


class _TaggedCompaniesParams(BaseModel):
    page: int = 1
    size: int = 10
    user_id: str | None = None


class TagRepository:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def apply_tag(self, company_id: str, payload: TagCreate) -> Tag:
        logger.debug(f"POST /companies/{company_id}/tags tag={payload.tag!r}")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/companies/{company_id}/tags",
                json=payload.model_dump(exclude_none=True, mode="json"),
            )
            response.raise_for_status()
            return Tag.model_validate(response.json())

    async def remove_tag(
        self,
        company_id: str,
        tag: str,
        tag_type: TagType,
        user_id: str | None = None,
    ) -> None:
        logger.debug(f"DELETE /companies/{company_id}/tags/{tag} type={tag_type}")
        params = _RemoveTagParams(tag_type=tag_type, user_id=user_id)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.delete(
                f"{self._base_url}/companies/{company_id}/tags/{tag}",
                params=params.model_dump(exclude_none=True, mode="json"),
            )
            response.raise_for_status()

    async def list_tags(self, user_id: str | None = None) -> list[TagSummary]:
        logger.debug(f"GET /tags user_id={user_id!r}")
        params = _ListTagsParams(user_id=user_id)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                f"{self._base_url}/tags",
                params=params.model_dump(exclude_none=True),
            )
            response.raise_for_status()
            return [TagSummary.model_validate(t) for t in response.json()]

    async def get_tagged_companies(
        self,
        tag: str,
        user_id: str | None = None,
        page: int = 1,
        size: int = 10,
    ) -> CompanyTagsResponse:
        logger.debug(f"GET /tags/{tag}/companies page={page}")
        params = _TaggedCompaniesParams(page=page, size=size, user_id=user_id)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                f"{self._base_url}/tags/{tag}/companies",
                params=params.model_dump(exclude_none=True),
            )
            response.raise_for_status()
            return CompanyTagsResponse.model_validate(response.json())
