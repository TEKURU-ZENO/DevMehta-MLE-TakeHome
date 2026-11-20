set shell := ["bash", "-euo", "pipefail", "-c"]

default: setup

# Install dependencies (producer, consumer, API, tests).
setup:
    uv sync --locked

# Start Postgres (with pgvector) and RabbitMQ, and wait until both are healthy.
up:
    docker compose up -d --wait

# Stop everything and delete the volumes.
down:
    docker compose down -v

# Replay the registry feed onto the company-events queue.
publish:
    uv run python -m producer.publish

# Run the consumer (Ctrl+C to stop). Creates the schema on start. The first run downloads the embedding model.
consume:
    uv run python -m similar.consumer

# Run the consumer until the queue has been empty for 5 seconds, then exit.
drain:
    uv run python -m similar.consumer --exit-when-idle 5

# Run the HTTP API on http://127.0.0.1:8000 (docs at /docs).
api:
    uv run uvicorn similar.api:app --host 127.0.0.1 --port 8000

# Every delivered message accounted for: applied, duplicate or quarantined.
report:
    uv run python -m similar.report

# All tests. The database tests use a separate companies_test database and need `just up`.
test:
    uv run pytest -q

# Compare embedding models on the feed (writes eval/results.md). No database needed.
eval:
    uv run python -m scripts.evaluate

# Latency of the similar-companies endpoint against a running `just api`.
loadtest:
    uv run python -m scripts.loadtest

# From an empty system to a working API: publish, consume, report.
demo: up publish drain report
