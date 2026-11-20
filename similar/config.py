"""Settings, read once from the environment. Defaults match docker-compose.yml.

Hosts are 127.0.0.1 rather than localhost: compose binds IPv4 only, and on Windows `localhost` tries ::1 first and
stalls for the full connect timeout before falling back.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    rabbitmq_url: str
    queue: str
    embedding_model: str
    embedding_dim: int
    prefetch: int
    workers: int


def load_settings() -> Settings:
    pg_port = os.environ.get("POSTGRES_PORT", "5432")
    mq_port = os.environ.get("RABBITMQ_PORT", "5672")
    return Settings(
        database_url=os.environ.get("DATABASE_URL", f"postgresql://postgres:postgres@127.0.0.1:{pg_port}/companies"),
        rabbitmq_url=os.environ.get("RABBITMQ_URL", f"amqp://guest:guest@127.0.0.1:{mq_port}/"),
        queue=os.environ.get("QUEUE", "company-events"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "384")),
        prefetch=int(os.environ.get("PREFETCH", "32")),
        workers=int(os.environ.get("WORKERS", "4")),
    )
