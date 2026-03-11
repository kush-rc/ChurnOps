"""API middleware for logging, timing, and request tracking."""

import time

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.metrics_collector import REQUEST_COUNT, REQUEST_LATENCY


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs all incoming requests with timing."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()

        response = await call_next(request)

        elapsed = time.perf_counter() - start

        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed:.3f}s)"
        )

        # Update Prometheus metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(elapsed)

        return response
