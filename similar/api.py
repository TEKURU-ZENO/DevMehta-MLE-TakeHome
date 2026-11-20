"""HTTP API for the "Similar companies" panel.

GET /companies/{registration_number}/similar?limit=4
GET /health

Every error has the same shape: {"error": {"code": "...", "message": "..."}}.

Usage: uv run uvicorn similar.api:app --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import psycopg
import psycopg_pool
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from similar.config import load_settings
from similar.db import create_pool
from similar.normalize import QuarantineError, normalize_registration_number

log = logging.getLogger("similar.api")

DEFAULT_LIMIT = 4
MAX_LIMIT = 20
HNSW_EF_SEARCH = 64

COMPANY_SQL = """
SELECT registration_number, status, company_name, nature_of_business, industry, state, updated_at,
       embedding IS NOT NULL AS has_embedding
FROM companies WHERE registration_number = %s
"""

# The inner query is the shape the HNSW index can serve: ORDER BY distance to a scalar subquery, LIMIT k, with the
# partial index's predicates. (Joining to the target row instead makes the planner fall back to a sequential scan.)
# The outer query only orders exact-score ties by registration number so the response is stable.
SIMILAR_SQL = """
SELECT registration_number, company_name, industry, state, 1 - distance AS score
FROM (
    SELECT c.registration_number, c.company_name, c.industry, c.state,
           c.embedding <=> (SELECT embedding FROM companies WHERE registration_number = %(reg)s) AS distance
    FROM companies c
    WHERE c.status = 'active' AND c.embedding IS NOT NULL AND c.registration_number <> %(reg)s
    ORDER BY distance
    LIMIT %(limit)s
) nearest
ORDER BY distance, registration_number
"""


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(database_url: str | None = None) -> FastAPI:
    database_url = database_url or load_settings().database_url

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Don't wait for the database: the service should start (and answer 503) even while Postgres is down.
        pool = create_pool(database_url, min_size=1, max_size=16, timeout=2.0, open_now=False)
        pool.open(wait=False)
        app.state.pool = pool
        try:
            yield
        finally:
            pool.close()

    app = FastAPI(title="Similar companies", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return error_response(400, "invalid_request", "; ".join(e.get("msg", "") for e in exc.errors()))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return error_response(exc.status_code, "not_found" if exc.status_code == 404 else "http_error", str(exc.detail))

    @app.exception_handler(psycopg.OperationalError)
    @app.exception_handler(psycopg_pool.PoolTimeout)
    async def _db_down(_: Request, exc: Exception):
        log.warning("database unavailable: %s", exc)
        return error_response(503, "database_unavailable", "The company database is unavailable. Try again shortly.")

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unexpected error")
        return error_response(500, "internal_error", "Something went wrong.")

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        with request.app.state.pool.connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok"}

    @app.get("/companies/{registration_number}/similar")
    def similar(registration_number: str, request: Request, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        try:
            reg = normalize_registration_number(registration_number)
        except QuarantineError:
            raise ApiError(400, "invalid_registration_number",
                           f"{registration_number!r} is not a registration number like SYN-001550") from None
        if not 1 <= limit <= MAX_LIMIT:
            raise ApiError(400, "invalid_limit", f"limit must be between 1 and {MAX_LIMIT}")

        with request.app.state.pool.connection() as conn, conn.transaction():
            company = conn.execute(COMPANY_SQL, (reg,)).fetchone()
            if company is None:
                raise ApiError(404, "company_not_found", f"No company with registration number {reg}")
            reg_, status, name, description, industry, state, updated_at, has_embedding = company
            if status == "struck_off":
                raise ApiError(410, "company_struck_off", f"{reg} has been struck off the register")

            body: dict[str, Any] = {
                "company": {
                    "registrationNumber": reg_, "companyName": name, "natureOfBusiness": description,
                    "industry": industry, "state": state, "updatedAt": updated_at.isoformat(),
                },
                "similar": [],
                "reason": None,
            }
            if not has_embedding:
                body["reason"] = "no_business_description"
                return body

            # iterative_scan keeps the index scan going if the partial-index filter or self-exclusion drops rows.
            conn.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}; SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = conn.execute(SIMILAR_SQL, {"reg": reg, "limit": limit}).fetchall()

        body["similar"] = [
            {"registrationNumber": r, "companyName": n, "industry": i, "state": s, "score": round(float(score), 4)}
            for r, n, i, s, score in rows
        ]
        if not rows:
            body["reason"] = "no_similar_companies"
        return body

    return app


app = create_app()
