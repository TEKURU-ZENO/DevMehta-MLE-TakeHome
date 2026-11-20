"""Print the accounting for the live database: every delivered message is either applied, a duplicate or quarantined.

Usage: uv run python -m similar.report
"""

import json

import psycopg

from similar.config import load_settings
from similar.db import accounting


def main() -> None:
    settings = load_settings()
    with psycopg.connect(settings.database_url) as conn:
        report = accounting(conn, settings.embedding_model)
    report["balanced"] = report["distinct_messages"] + report["duplicate_deliveries"] == report["deliveries"] and (
        report["applied"] + report["quarantined"] == report["distinct_messages"])
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
