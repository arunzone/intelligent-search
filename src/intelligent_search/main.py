"""FastAPI application factory."""

import sys
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from intelligent_search.api.dependencies import get_agent_graph
from intelligent_search.api.router import router
from intelligent_search.api.tags_router import router as tags_router
from intelligent_search.config import get_settings
from intelligent_search.telemetry import configure_telemetry, instrument_app


def _configure_logging() -> None:
    """Configure loguru for Docker: verbose, full tracebacks, no buffering."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=get_settings().log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=True,  # full call chain on exceptions
        diagnose=True,  # variable values in tracebacks
        colorize=False,  # plain text — better for docker logs / log shippers
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    # ── startup ──────────────────────────────────────────────────────────────
    _configure_logging()
    logger.info("Starting Intelligent Search service")
    try:
        get_agent_graph().get_graph()
        logger.info("Agent graph compiled and ready")
    except Exception:
        # Log full traceback but keep the server alive so docker logs show the cause
        logger.error(
            "Agent graph compilation failed — service will retry on first request\n"
            + traceback.format_exc()
        )

    yield

    # ── shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down Intelligent Search service")


def create_app() -> FastAPI:
    configure_telemetry()

    app = FastAPI(
        title="Intelligent Company Search",
        description="Natural language search over the company database via a LangGraph agent.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            f"Unhandled error on {request.method} {request.url}\n"
            + traceback.format_exc()
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
        )

    app.include_router(router)
    app.include_router(tags_router)
    instrument_app(app)
    return app


app = create_app()
