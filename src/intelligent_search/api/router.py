"""FastAPI router — exposes the intelligent search endpoint."""

from fastapi import APIRouter, Depends
from loguru import logger

from intelligent_search.api.dependencies import get_search_service
from intelligent_search.application.search_service import SearchService
from intelligent_search.domain.models import (
    IntelligentSearchRequest,
    IntelligentSearchResponse,
)

router = APIRouter(prefix="/search", tags=["intelligent-search"])


@router.post("/intelligent", response_model=IntelligentSearchResponse)
async def intelligent_search(
    request: IntelligentSearchRequest,
    service: SearchService = Depends(get_search_service),
) -> IntelligentSearchResponse:
    """
    Accept a natural language query and return company results.

    Examples:
    - "tech companies in California"
    - "small software startups founded after 2015"
    - "healthcare companies in Germany with more than 1000 employees"
    """
    logger.info(
        f"POST /search/intelligent query={request.query!r} industry={request.industry!r}"
    )
    return await service.search(request)
