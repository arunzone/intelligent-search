# Intelligent Company Search

## Project Overview

A two-service system built on a 7M-company dataset that delivers both structured and
natural-language company search.

- **Part 1** (`port 8000`) — Structured search API backed by OpenSearch: full-text and
  filtered search across name, industry, location, size, and founding year.
- **Part 2** (`port 8001`) — Intelligent search layer: a LangGraph agent that translates
  natural language into precise search parameters, with LLM-driven semantic matching.
- **Part 3** — Tagging system proxied through Part 2, enabling personal and shared
  company labels.

---

## Setup

### Prerequisites

- Docker + Docker Compose
- (Optional) OpenAI API key — falls back to local Ollama if unset

### Running locally

```bash
cp .env.example .env
# Optionally set OPENAI_API_KEY in .env for OpenAI as primary LLM

docker compose up --build
```

| Service | URL |
|---|---|
| Intelligent Search API | http://localhost:8001 |
| API docs (Swagger) | http://localhost:8001/docs |
| Metrics (Prometheus) | http://localhost:8001/metrics |
| Health check | http://localhost:8001/health |

### Development (hot reload)

```bash
docker compose -f docker-compose.dev.yml up --build
```

### Running tests

```bash
docker-compose -f docker-compose.dev.yml run --rm app tox
```

---

## Architecture Overview

```
Client
  │
  ▼
Part 2 — Intelligent Search API  (FastAPI · port 8001)
  │
  ├── SearchService (application layer)
  │     ├── Direct path  ──────────────────────────────────────┐
  │     └── Agent path                                         │
  │           └── LangGraph StateGraph                         │
  │                 ├── agent node  (OpenAI → Ollama fallback) │
  │                 └── tools node  (search_companies)         │
  │                       │                                    │
  │                       ▼                                    ▼
  └──────────────── Part 1 — Company Search API  (port 8000)
                          │
                          └── OpenSearch  (7M company index)
```

**Layers within Part 2:**

| Layer | Module | Responsibility |
|---|---|---|
| API | `api/router.py`, `api/tags_router.py` | HTTP in/out, validation |
| Application | `application/search_service.py` | Orchestration, caching, routing |
| Agent | `agent/graph.py`, `agent/tools.py` | LLM reasoning, tool execution |
| Infrastructure | `infrastructure/company_search_repository.py` | HTTP → Part 1 |
| Domain | `domain/models.py` | Pure data contracts, no side effects |

---

## Design Decisions

**LangGraph agent over a single LLM call**
A single prompt-to-answer approach can't reliably extract structured filters and handle
multi-step reasoning. LangGraph's `StateGraph` gives explicit control over the
agent→tools→agent loop, making behaviour inspectable and testable. The tool call args
are also extracted post-invocation and cached as resolved `CompanySearchParams` — so
subsequent paginated requests bypass the LLM entirely.

**OpenAI primary / Ollama fallback**
LangChain's `with_fallbacks()` provides resilience with zero branching logic. Locally,
Ollama runs without an API key; in production, OpenAI is the default. Switching is a
single env var (`OPENAI_API_KEY`).

**Query routing in `SearchService`**
Three paths, decided upfront: cache hit → paginate; no query text → direct filter search;
query present → agent. This avoids LLM cost on pure-filter and repeat requests.

**Prompt design — conservative extraction**
The system prompt instructs the agent to *never* infer or assume filter values, only map
explicit mentions. This prevents hallucinated filters (e.g., assuming "California" means
"United States") and keeps results trustworthy.

**Tagging as a proxy, not a store**
Part 2 exposes the full tagging API but delegates persistence to Part 1. This keeps Part 2
stateless and avoids duplicating storage concerns.

---

## Implementation Highlights

- **`SearchService.search(request)`** — Single entry point accepting the full request
  object; eliminates repeated parameter threading through internal methods.
- **Result cache** — `dict[str, CompanySearchParams]` keyed by
  `request.model_dump_json(exclude={page, size, sort, sort_order})`. Subsequent pages of
  the same logical search re-use the agent-resolved params. Capped at 256 entries with
  FIFO eviction.
- **`_extract_resolved_params`** — After agent invocation, the last `search_companies`
  tool call args are recovered from the message history and merged with request-level
  fallbacks. This is what gets cached — not the response, but the *parameters*.
- **`SearchCompaniesInput` validators** — Pydantic `field_validator` coerces LLM list
  outputs and empty strings to `None`, guarding against common LLM serialisation quirks.
- **Retry with backoff** — `tenacity` wraps the Part 1 HTTP call: 3 attempts, exponential
  backoff (1–4 s). Handles transient failures transparently.
- **Observability** — OpenTelemetry traces (console exporter) + Prometheus metrics at
  `/metrics`. All inbound (FastAPI) and outbound (httpx) calls auto-instrumented. Health
  check at `/health`.

---

## Engineering Best Practices

- **SRP** — Each class has one job: `SearchService` orchestrates; repositories fetch;
  `SearchAgentGraph` compiles the graph; `build_model` constructs the LLM chain.
- **DIP** — All dependencies constructor-injected; `dependencies.py` is the single wiring
  point. Tests override via FastAPI's `dependency_overrides`.
- **OCP** — Sort strategy uses a dict of key functions (`_NUMERIC_SORT_KEYS`); adding a
  new sort field requires no branching logic changes.
- **KISS** — The router's handler is three lines. The agent has two nodes. No framework
  beyond what's needed.
- **Clean architecture** — Dependency direction flows inward: API → Application → Domain.
  Infrastructure depends on Domain; nothing in Domain imports from other layers.

---

## Scalability and Performance

| Concern | Approach |
|---|---|
| LLM latency | Cache resolved params — repeated/paginated calls skip the LLM |
| LLM availability | OpenAI → Ollama fallback via `with_fallbacks()` |
| Part 1 resilience | Tenacity retry with exponential backoff |
| Stateless service | No shared mutable state between requests; safe to run multiple replicas |
| 30 RPS LLM target | Horizontal scaling of Part 2 replicas behind a load balancer; cache hit rate reduces effective LLM load significantly |

**Scaling to 10x load:** Part 2 scales horizontally (fully stateless). The in-process
cache should be replaced with Redis to share resolved params across replicas. Part 1 and
OpenSearch scale independently via replica shards and read replicas.

---

## Future Improvements

- **Distributed cache** — Replace the in-process dict with Redis for cross-replica cache
  sharing and TTL-based expiry.
- **Async httpx client lifecycle** — Share a persistent `AsyncClient` per repository
  (created on startup, closed on shutdown) for connection pooling.
- **Streaming responses** — Stream agent token output for lower perceived latency on
  long queries.
- **Agentic web search** — Extend the tool set with a web search tool to answer queries
  like "companies that announced funding in the last 2 months".
- **Structured tracing** — Replace the console span exporter with OTLP export to a
  collector (Jaeger / Tempo) for distributed trace correlation across Part 1 and Part 2.
- **Rate limiting** — Add a middleware layer (e.g., `slowapi`) to enforce per-user LLM
  request limits and protect against runaway costs.
