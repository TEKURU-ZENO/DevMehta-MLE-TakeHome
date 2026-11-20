"""Database fixtures. Tests that need Postgres run against a separate `companies_test` database on the compose
Postgres (or TEST_DATABASE_URL) and are skipped with a clear message if it isn't reachable."""

import os
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from similar.db import Processor, create_pool, init_schema
from similar.embedding import HashEmbedder

ADMIN_URL = os.environ.get(
    "TEST_ADMIN_DATABASE_URL", f"postgresql://postgres:postgres@127.0.0.1:{os.environ.get('POSTGRES_PORT', '5432')}/postgres"
)
TEST_DB = "companies_test"
TABLES = "quarantined_events, company_events, companies, embedding_cache"


def _test_url() -> str:
    if url := os.environ.get("TEST_DATABASE_URL"):
        return url
    return urlunparse(urlparse(ADMIN_URL)._replace(path=f"/{TEST_DB}"))


@pytest.fixture(scope="session")
def database_url() -> str:
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
            conn.execute(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable ({exc}). Run `just up` first.")
    url = _test_url()
    init_schema(url, 384)
    return url


@pytest.fixture(scope="session")
def pool(database_url):
    pool = create_pool(database_url, max_size=8)
    yield pool
    pool.close()


@pytest.fixture
def clean_db(pool):
    with pool.connection() as conn:
        conn.execute(f"TRUNCATE {TABLES}")
    return pool


@pytest.fixture
def processor(clean_db) -> Processor:
    return Processor(clean_db, HashEmbedder(384))
