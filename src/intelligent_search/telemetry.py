"""OpenTelemetry setup — traces via console, metrics via Prometheus /metrics endpoint."""

from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


def configure_telemetry(service_name: str = "intelligent-search") -> None:
    """Configure trace and metric providers.

    Traces  — printed to stdout as each span completes (BatchSpanProcessor).
    Metrics — exposed at GET /metrics in Prometheus format (PrometheusMetricReader).
    httpx   — all outbound HTTP calls auto-instrumented (tag + company search repos).
    FastAPI — all inbound HTTP requests auto-instrumented (latency, status, route).
    """
    resource = Resource(attributes={SERVICE_NAME: service_name})

    # ── Traces ────────────────────────────────────────────────────────────────
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    # ── Metrics ───────────────────────────────────────────────────────────────
    # PrometheusMetricReader registers metrics with the prometheus_client registry;
    # the /metrics route serves them via make_asgi_app().
    reader = PrometheusMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    # ── Auto-instrumentation ──────────────────────────────────────────────────
    HTTPXClientInstrumentor().instrument()


def instrument_app(app: object) -> None:
    """Attach FastAPI instrumentation and mount /health + /metrics after app creation."""
    from fastapi import FastAPI
    from prometheus_client import make_asgi_app

    assert isinstance(app, FastAPI)

    FastAPIInstrumentor.instrument_app(app)

    @app.get("/health", tags=["ops"], include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.mount("/metrics", make_asgi_app())
