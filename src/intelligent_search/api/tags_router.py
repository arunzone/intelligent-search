"""FastAPI router for tag management — proxies to the Part 1 tag API."""

from typing import Optional

from fastapi import APIRouter, Depends, Query, Response

from intelligent_search.api.dependencies import get_tag_repository
from intelligent_search.domain.models import (
    CompanyTagsResponse,
    Tag,
    TagCreate,
    TagSummary,
    TagType,
)
from intelligent_search.infrastructure.tag_repository import TagRepository

router = APIRouter(tags=["tags"])


@router.post("/companies/{company_id}/tags", response_model=Tag, status_code=201)
async def apply_tag(
    company_id: str,
    payload: TagCreate,
    repo: TagRepository = Depends(get_tag_repository),
) -> Tag:
    """Apply a tag to a company. Idempotent — applying the same tag twice has no effect."""
    return await repo.apply_tag(company_id, payload)


@router.delete("/companies/{company_id}/tags/{tag}", status_code=204)
async def remove_tag(
    company_id: str,
    tag: str,
    tag_type: TagType = Query(..., description="Tag type: public or personal"),
    user_id: Optional[str] = Query(
        default=None, description="User ID (required for personal tags)"
    ),
    repo: TagRepository = Depends(get_tag_repository),
) -> Response:
    """Remove a tag from a company."""
    await repo.remove_tag(company_id, tag, tag_type, user_id)
    return Response(status_code=204)


@router.get("/tags", response_model=list[TagSummary])
async def list_tags(
    user_id: Optional[str] = Query(
        default=None, description="User ID — includes personal tags when provided"
    ),
    repo: TagRepository = Depends(get_tag_repository),
) -> list[TagSummary]:
    """List unique tags with company counts."""
    return await repo.list_tags(user_id)


@router.get("/tags/{tag}/companies", response_model=CompanyTagsResponse)
async def get_tagged_companies(
    tag: str,
    user_id: Optional[str] = Query(
        default=None, description="User ID — includes personal tags when provided"
    ),
    page: int = Query(default=1, ge=1, description="Page number"),
    size: int = Query(default=10, ge=1, le=100, description="Results per page"),
    repo: TagRepository = Depends(get_tag_repository),
) -> CompanyTagsResponse:
    """List company IDs carrying a specific tag, paginated."""
    return await repo.get_tagged_companies(tag, user_id, page, size)
