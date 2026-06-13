# src/api/routes/metrics.py
"""Prometheus metrics endpoint."""

import os

from fastapi import APIRouter
from fastapi.responses import Response

# from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess

router = APIRouter(tags=["observability"])


# @router.get("/metrics")
@router.get("/metrics", include_in_schema=False)  # cache de Swagger
async def metrics() -> Response:
    """
    Expose Prometheus metrics in text format.

    Scraped by Prometheus server every 15-30 seconds.
    Format: https://prometheus.io/docs/instrumenting/exposition_formats/
    """
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        # Production case: multiple workers (gunicorn), aggregate counters
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        # Dev/single-worker uvicorn: use the default registry
        data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
