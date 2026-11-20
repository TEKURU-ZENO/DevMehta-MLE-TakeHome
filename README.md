# Similar companies: consumer and API

My solution to the OneCredit MLE take-home. The original brief is in [`BRIEF.md`](BRIEF.md). Decisions,
assumptions, the model comparison, latency numbers and what's left are in **[`DESIGN.md`](DESIGN.md)**.

## Run it

You need Docker, [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just), as in the brief.

```bash
just setup      # install dependencies
just up         # start Postgres (pgvector) and RabbitMQ
just publish    # publish the feed (2,518 messages)
just drain      # run the consumer until the queue is empty, then exit
just report     # accounting: delivered = applied + duplicates + quarantined
just api        # start the API on http://127.0.0.1:8000 (Ctrl+C to stop)
```

`just demo` runs `up`, `publish`, `drain` and `report` in one go. To keep the consumer running instead, use
`just consume`. The consumer creates its tables on start. The first run downloads the embedding model (about
220 MB) into fastembed's cache.

Then:

```bash
curl http://127.0.0.1:8000/companies/SYN-001550/similar
curl "http://127.0.0.1:8000/companies/syn-001550/similar?limit=8"   # IDs are normalised like the feed
curl http://127.0.0.1:8000/companies/SYN-001748/similar             # struck off -> 410
```

Interactive docs are at http://127.0.0.1:8000/docs.

## Tests and measurements

```bash
just test       # 88 tests; the database ones need `just up` and use a separate companies_test database (~6 min)
just eval       # compare embedding models on the feed -> eval/results.md (no database needed)
just loadtest   # latency against a running `just api`
```

## Configuration

Environment variables, with defaults that match `docker-compose.yml`:

| Variable | Default |
|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@127.0.0.1:${POSTGRES_PORT:-5432}/companies` |
| `RABBITMQ_URL` | `amqp://guest:guest@127.0.0.1:${RABBITMQ_PORT:-5672}/` |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `EMBEDDING_DIM` | `384` |
| `PREFETCH`, `WORKERS` | `32`, `4` |

The defaults use `127.0.0.1`, not `localhost`. Compose binds IPv4 only, and on Windows `localhost` tries `::1`
first and waits for the full connect timeout.

## Layout

```
similar/
  normalize.py   one pure function per field; parse_message() -> Event or QuarantineError(reason)
  fold.py        the "latest data wins" rule in plain Python (the reference the tests compare against)
  db.py          schema, and Processor.handle(): one message -> one transaction
  consumer.py    RabbitMQ consumer: ack after commit, backoff on transient errors, quarantine poison messages
  embedding.py   fastembed model (and a hashed bag-of-words used only by tests)
  api.py         FastAPI app: GET /companies/{id}/similar, GET /health
  report.py      accounting for the live database
scripts/
  evaluate.py    model comparison (P@4, nDCG@4, Malay->English concept accuracy, confusions, failures)
  loadtest.py    client-side latency percentiles
tests/           unit, replay/accounting, consumer ack, API contract
eval/            results of the last `just eval`
```
