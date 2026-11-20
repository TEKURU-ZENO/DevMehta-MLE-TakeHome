"""The consumer's write path against a real Postgres: replay the whole feed in different orders and check the outcome
is identical, fully accounted for, and equal to the pure-Python reference (similar.fold)."""

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from similar.db import Outcome, Processor, accounting
from similar.embedding import HashEmbedder
from similar.fold import fold
from similar.normalize import QuarantineError, parse_message

FEED_LINES = (Path(__file__).parent.parent / "data" / "feed.jsonl").read_text(encoding="utf-8").splitlines()
SNAPSHOT_SQL = """
SELECT registration_number, status, updated_at, last_message_id, company_name, nature_of_business, state, industry,
       text_hash, embedding::text
FROM companies ORDER BY registration_number
"""


def mid(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()  # what producer/publish.py sets as message_id


def replay(processor: Processor, lines: list[str], workers: int = 1) -> None:
    bodies = [line.encode() for line in lines]
    if workers == 1:
        for b in bodies:
            processor.handle(b, mid(b))
    else:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda b: processor.handle(b, mid(b)), bodies))


def snapshot(pool) -> list[tuple]:
    with pool.connection() as conn:
        return conn.execute(SNAPSHOT_SQL).fetchall()


def reset(pool) -> None:
    with pool.connection() as conn:
        conn.execute("TRUNCATE quarantined_events, company_events, companies, embedding_cache")


def report(processor: Processor) -> dict:
    with processor.pool.connection() as conn:
        return accounting(conn, processor.embedder.name)


@pytest.fixture(scope="module")
def in_order_snapshot(pool):
    reset(pool)
    replay(Processor(pool, HashEmbedder(384)), FEED_LINES)
    return snapshot(pool)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_shuffled_replay_gives_identical_state(processor, in_order_snapshot, seed):
    lines = FEED_LINES[:]
    random.Random(seed).shuffle(lines)
    reset(processor.pool)  # the module fixture may have filled the database after this test's own reset
    replay(processor, lines)
    assert snapshot(processor.pool) == in_order_snapshot


def test_reversed_concurrent_replay_gives_identical_state(processor, in_order_snapshot):
    reset(processor.pool)
    replay(processor, FEED_LINES[::-1], workers=8)
    assert snapshot(processor.pool) == in_order_snapshot


def test_replaying_twice_changes_nothing_and_counts_duplicates(processor, in_order_snapshot):
    reset(processor.pool)
    replay(processor, FEED_LINES)
    replay(processor, FEED_LINES)
    assert snapshot(processor.pool) == in_order_snapshot
    r = report(processor)
    assert r["deliveries"] == 2 * 2518
    assert r["duplicate_deliveries"] == 2518 + 68


def test_exact_accounting(processor):
    replay(processor, FEED_LINES)
    r = report(processor)
    assert r["deliveries"] == 2518
    assert r["distinct_messages"] + r["duplicate_deliveries"] == 2518
    assert r["distinct_messages"] == r["applied"] + r["quarantined"]
    assert r["quarantined"] == 32
    assert r["companies"] == 2000
    assert r["struck_off"] == 8
    assert r["active"] == 1992
    assert r["quarantine_reasons"] == {
        "invalid_json": 5, "missing_updated_at": 7, "missing_registration_number": 7, "missing_data": 5,
        "invalid_data": 4, "invalid_registration_number": 3, "not_an_object": 1,
    }


def test_database_matches_reference_fold(processor):
    replay(processor, FEED_LINES)
    events = []
    for line in dict.fromkeys(FEED_LINES):  # distinct bodies, same as distinct message ids
        try:
            events.append((mid(line.encode()), parse_message(line.encode())))
        except QuarantineError:
            pass
    expected = fold(events)
    rows = snapshot(processor.pool)
    assert len(rows) == len(expected)
    for reg, status, updated_at, last_mid, name, nob, state, industry, text_hash, _ in rows:
        e = expected[reg]
        assert (status, updated_at, last_mid, name, nob, state, industry, text_hash) == (
            e.status, e.updated_at, e.last_message_id, e.company_name, e.nature_of_business, e.state, e.industry,
            e.text_hash)


# --- targeted cases ------------------------------------------------------------------------------------------------


def msg(reg="SYN-000001", updated_at="2026-08-14T09:00:00+08:00", type_="company-upserted", **data) -> bytes:
    payload = {"type": type_, "registrationNumber": reg, "updatedAt": updated_at}
    if type_ == "company-upserted":
        payload["data"] = {"companyName": "Kopi Co", "natureOfBusiness": "Coffee shop", "state": "Johor",
                           "industry": "Food & Beverage"} | data
    return json.dumps(payload).encode()


def company(pool, reg="SYN-000001"):
    with pool.connection() as conn:
        return conn.execute(
            "SELECT status, nature_of_business, state, embedding IS NOT NULL FROM companies WHERE registration_number = %s",
            (reg,),
        ).fetchone()


def apply(processor, body: bytes) -> Outcome:
    return processor.handle(body, mid(body))


def test_out_of_order_update(processor):
    apply(processor, msg(updated_at="2026-08-14T10:00:00+08:00", natureOfBusiness="Bakery"))
    apply(processor, msg(updated_at="2026-08-14T09:00:00+08:00", natureOfBusiness="Coffee shop"))
    assert company(processor.pool)[1] == "Bakery"


def test_old_update_after_tombstone_stays_struck_off(processor):
    apply(processor, msg(updated_at="2026-08-14T12:00:00+08:00", type_="company-struck-off"))
    apply(processor, msg(updated_at="2026-08-14T09:00:00+08:00"))
    assert company(processor.pool)[0] == "struck_off"


def test_lowercase_and_padded_id_is_the_same_company(processor):
    apply(processor, msg(reg=" syn-000001 ", updated_at="2026-08-14T09:00:00+08:00", state="Johor"))
    apply(processor, msg(reg="SYN-000001", updated_at="2026-08-14T10:00:00+08:00", state="Perak"))
    with processor.pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM companies").fetchone()[0] == 1
    assert company(processor.pool)[2] == "Perak"


def test_unix_millis_compared_with_iso(processor):
    apply(processor, msg(updated_at=1786670837000, natureOfBusiness="Newer"))  # 09:27:17 +08:00
    apply(processor, msg(updated_at="2026-08-14T09:27:00+08:00", natureOfBusiness="Older"))
    assert company(processor.pool)[1] == "Newer"


def test_missing_description_is_stored_without_embedding(processor):
    assert apply(processor, msg(natureOfBusiness="")) is Outcome.APPLIED
    assert company(processor.pool) == ("active", None, "Johor", False)


def test_broken_json_is_quarantined_and_redelivery_is_a_duplicate(processor):
    broken = b'{"type": "company-upserted", "registrationNumber": "SYN-0'
    assert apply(processor, broken) is Outcome.QUARANTINED
    assert apply(processor, broken) is Outcome.DUPLICATE
    assert report(processor)["quarantine_reasons"] == {"invalid_json": 1}


def test_state_change_does_not_re_embed(clean_db):
    calls = []

    class CountingEmbedder(HashEmbedder):
        def embed(self, texts):
            calls.extend(texts)
            return super().embed(texts)

    processor = Processor(clean_db, CountingEmbedder(384))
    apply(processor, msg(updated_at="2026-08-14T09:00:00+08:00", state="Johor"))
    apply(processor, msg(updated_at="2026-08-14T10:00:00+08:00", state="Perak"))
    apply(processor, msg(reg="SYN-000002"))  # another company, same description
    assert calls == ["Coffee shop"]
