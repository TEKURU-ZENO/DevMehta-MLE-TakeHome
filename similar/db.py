"""Schema and the write path: one message in, one transaction out.

Three layers:
    company_events      every distinct message as received (raw bytes + normalised columns), keyed by message_id
    quarantined_events  the subset of company_events that can never be applied, with a reason code
    companies           one row per registration number: the current truth, rebuilt from company_events
plus embedding_cache, so a description is embedded once per model however many companies or events share it.
"""

from enum import StrEnum

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from similar.embedding import Embedder
from similar.normalize import Event, QuarantineError, Reason, message_id_for, parse_message


def schema_sql(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS company_events (
    message_id          text COLLATE "C" PRIMARY KEY,
    body                bytea       NOT NULL,
    first_received_at   timestamptz NOT NULL DEFAULT now(),
    delivery_count      integer     NOT NULL DEFAULT 1,
    outcome             text        NOT NULL CHECK (outcome IN ('applied', 'quarantined')),
    registration_number text,
    event_type          text,
    updated_at          timestamptz,
    company_name        text,
    nature_of_business  text,
    state               text,
    industry            text,
    text_hash           text,
    extra               jsonb
);
CREATE INDEX IF NOT EXISTS company_events_by_company
    ON company_events (registration_number, updated_at DESC) WHERE outcome = 'applied';

CREATE TABLE IF NOT EXISTS quarantined_events (
    message_id     text COLLATE "C" PRIMARY KEY REFERENCES company_events (message_id),
    reason         text        NOT NULL,
    detail         text        NOT NULL,
    quarantined_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    model     text NOT NULL,
    text_hash text NOT NULL,
    embedding vector({dim}) NOT NULL,
    PRIMARY KEY (model, text_hash)
);

CREATE TABLE IF NOT EXISTS companies (
    registration_number text PRIMARY KEY,
    status              text        NOT NULL CHECK (status IN ('active', 'struck_off')),
    updated_at          timestamptz NOT NULL,
    last_message_id     text COLLATE "C" NOT NULL,
    company_name        text,
    nature_of_business  text,
    state               text,
    industry            text,
    text_hash           text,
    embedding           vector({dim}),
    refreshed_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS companies_embedding_hnsw
    ON companies USING hnsw (embedding vector_cosine_ops)
    WHERE status = 'active' AND embedding IS NOT NULL;
"""


# Rebuild one company from its events. Same ordering as similar.fold: updatedAt, then struck-off beats upsert on a
# tie, then message_id. Status comes from the newest event, the card from the newest upsert.
REFRESH_COMPANY_SQL = """
WITH events AS (
    SELECT * FROM company_events WHERE registration_number = %(reg)s AND outcome = 'applied'
), latest AS (
    SELECT * FROM events
    ORDER BY updated_at DESC, (event_type = 'company-struck-off') DESC, message_id DESC LIMIT 1
), card AS (
    SELECT * FROM events WHERE event_type = 'company-upserted'
    ORDER BY updated_at DESC, message_id DESC LIMIT 1
)
INSERT INTO companies AS c (registration_number, status, updated_at, last_message_id, company_name,
                            nature_of_business, state, industry, text_hash, embedding)
SELECT l.registration_number,
       CASE WHEN l.event_type = 'company-struck-off' THEN 'struck_off' ELSE 'active' END,
       l.updated_at, l.message_id, k.company_name, k.nature_of_business, k.state, k.industry, k.text_hash,
       e.embedding
FROM latest l
LEFT JOIN card k ON true
LEFT JOIN embedding_cache e ON e.model = %(model)s AND e.text_hash = k.text_hash
ON CONFLICT (registration_number) DO UPDATE SET
    status = EXCLUDED.status,
    updated_at = EXCLUDED.updated_at,
    last_message_id = EXCLUDED.last_message_id,
    company_name = EXCLUDED.company_name,
    nature_of_business = EXCLUDED.nature_of_business,
    state = EXCLUDED.state,
    industry = EXCLUDED.industry,
    text_hash = EXCLUDED.text_hash,
    embedding = EXCLUDED.embedding,
    refreshed_at = now()
WHERE (c.status, c.updated_at, c.last_message_id, c.company_name, c.nature_of_business, c.state, c.industry,
       c.text_hash, c.embedding IS NULL)
      IS DISTINCT FROM
      (EXCLUDED.status, EXCLUDED.updated_at, EXCLUDED.last_message_id, EXCLUDED.company_name,
       EXCLUDED.nature_of_business, EXCLUDED.state, EXCLUDED.industry, EXCLUDED.text_hash, EXCLUDED.embedding IS NULL)
"""

INSERT_EVENT_SQL = """
INSERT INTO company_events (message_id, body, outcome, registration_number, event_type, updated_at, company_name,
                            nature_of_business, state, industry, text_hash, extra)
VALUES (%(message_id)s, %(body)s, %(outcome)s, %(reg)s, %(event_type)s, %(updated_at)s, %(company_name)s,
        %(nature_of_business)s, %(state)s, %(industry)s, %(text_hash)s, %(extra)s)
ON CONFLICT (message_id) DO UPDATE SET delivery_count = company_events.delivery_count + 1
RETURNING (xmax = 0) AS inserted
"""


class Outcome(StrEnum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"
    QUARANTINED = "quarantined"


def create_pool(database_url: str, *, min_size: int = 1, max_size: int = 8, timeout: float = 5.0,
                open_now: bool = True) -> ConnectionPool:
    return ConnectionPool(database_url, min_size=min_size, max_size=max_size, timeout=timeout,
                          configure=register_vector, open=open_now)


def init_schema(database_url: str, dim: int) -> None:
    # Separate connection: register_vector needs the extension to exist first.
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(schema_sql(dim))


class Processor:
    """Applies one message per call, all inside one transaction. Safe to call again with the same message."""

    def __init__(self, pool: ConnectionPool, embedder: Embedder):
        self.pool = pool
        self.embedder = embedder

    def handle(self, body: bytes, amqp_message_id: str | None = None) -> Outcome:
        message_id = message_id_for(body, amqp_message_id)
        try:
            event = parse_message(body)
        except QuarantineError as exc:
            return self.quarantine(body, message_id, exc.reason, exc.detail)

        # Embed outside the transaction (it's the slow part) and only for text this model hasn't seen before.
        vector = self._embedding_if_new(event)

        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(INSERT_EVENT_SQL, _event_params(message_id, body, event)).fetchone()
            if not row[0]:
                return Outcome.DUPLICATE
            if vector is not None:
                conn.execute(
                    "INSERT INTO embedding_cache (model, text_hash, embedding) VALUES (%s, %s, %s)"
                    " ON CONFLICT DO NOTHING",
                    (self.embedder.name, event.text_hash, Vector(vector)),
                )
            # Serialise work per company so two consumers can't each miss the other's event.
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (event.registration_number,))
            conn.execute(REFRESH_COMPANY_SQL, {"reg": event.registration_number, "model": self.embedder.name})
        return Outcome.APPLIED

    def quarantine(self, body: bytes, message_id: str, reason: Reason, detail: str) -> Outcome:
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(INSERT_EVENT_SQL, _event_params(message_id, body, None)).fetchone()
            if not row[0]:
                return Outcome.DUPLICATE
            conn.execute(
                "INSERT INTO quarantined_events (message_id, reason, detail) VALUES (%s, %s, %s)",
                (message_id, str(reason), detail[:2000]),
            )
        return Outcome.QUARANTINED

    def _embedding_if_new(self, event: Event) -> list[float] | None:
        if event.nature_of_business is None:
            return None
        with self.pool.connection() as conn:
            cached = conn.execute(
                "SELECT 1 FROM embedding_cache WHERE model = %s AND text_hash = %s",
                (self.embedder.name, event.text_hash),
            ).fetchone()
        if cached:
            return None
        return self.embedder.embed([event.nature_of_business])[0]


def _event_params(message_id: str, body: bytes, event: Event | None) -> dict:
    return {
        "message_id": message_id,
        "body": body,
        "outcome": "applied" if event else "quarantined",
        "reg": event.registration_number if event else None,
        "event_type": str(event.event_type) if event else None,
        "updated_at": event.updated_at if event else None,
        "company_name": event.company_name if event else None,
        "nature_of_business": event.nature_of_business if event else None,
        "state": event.state if event else None,
        "industry": event.industry if event else None,
        "text_hash": event.text_hash if event else None,
        "extra": Jsonb(event.extra) if event and event.extra else None,
    }


ACCOUNTING_SQL = """
SELECT
    (SELECT coalesce(sum(delivery_count), 0) FROM company_events)                      AS deliveries,
    (SELECT count(*) FROM company_events)                                               AS distinct_messages,
    (SELECT coalesce(sum(delivery_count - 1), 0) FROM company_events)                   AS duplicate_deliveries,
    (SELECT count(*) FROM company_events WHERE outcome = 'applied')                     AS applied,
    (SELECT count(*) FROM quarantined_events)                                           AS quarantined,
    (SELECT count(*) FROM companies)                                                    AS companies,
    (SELECT count(*) FROM companies WHERE status = 'active')                            AS active,
    (SELECT count(*) FROM companies WHERE status = 'struck_off')                        AS struck_off,
    (SELECT count(*) FROM companies WHERE status = 'active' AND embedding IS NULL)      AS active_without_embedding,
    (SELECT count(*) FROM embedding_cache WHERE model = %s)                             AS embedded_texts
"""


def accounting(conn: psycopg.Connection, model: str) -> dict:
    cur = conn.execute(ACCOUNTING_SQL, (model,))
    names = [d.name for d in cur.description]
    report = dict(zip(names, cur.fetchone(), strict=True))
    report["quarantine_reasons"] = dict(
        conn.execute("SELECT reason, count(*) FROM quarantined_events GROUP BY reason ORDER BY 2 DESC").fetchall()
    )
    return report
