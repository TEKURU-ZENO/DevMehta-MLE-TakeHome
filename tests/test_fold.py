"""The "latest data wins" rule in isolation, plus the feed-level facts it should produce."""

import hashlib
import random
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from similar.fold import fold
from similar.normalize import Event, EventType, QuarantineError, parse_message

T0 = datetime(2026, 8, 14, 1, tzinfo=UTC)
FEED = Path(__file__).parent.parent / "data" / "feed.jsonl"


def upsert(minutes: int, description: str, reg: str = "SYN-000001") -> Event:
    return Event(reg, EventType.UPSERTED, T0 + timedelta(minutes=minutes), company_name="Co",
                 nature_of_business=description)


def struck_off(minutes: int, reg: str = "SYN-000001") -> Event:
    return Event(reg, EventType.STRUCK_OFF, T0 + timedelta(minutes=minutes))


def test_newer_update_wins_whatever_the_arrival_order():
    events = [("a", upsert(0, "old")), ("b", upsert(10, "new"))]
    assert fold(events)["SYN-000001"].nature_of_business == "new"
    assert fold(events[::-1])["SYN-000001"].nature_of_business == "new"


def test_old_upsert_after_tombstone_does_not_revive():
    state = fold([("a", upsert(0, "cafe")), ("t", struck_off(10)), ("late", upsert(5, "stale"))])["SYN-000001"]
    assert state.status == "struck_off"
    assert state.nature_of_business == "stale"  # newest upsert still provides the last known card


def test_newer_upsert_after_tombstone_reactivates():
    assert fold([("t", struck_off(10)), ("b", upsert(20, "back"))])["SYN-000001"].status == "active"


def test_tombstone_wins_an_exact_tie():
    for events in ([("z", upsert(10, "x")), ("a", struck_off(10))], [("a", upsert(10, "x")), ("z", struck_off(10))]):
        assert fold(events)["SYN-000001"].status == "struck_off"


def test_message_id_breaks_ties_between_upserts():
    events = [("m1", upsert(10, "first")), ("m2", upsert(10, "second"))]
    assert fold(events)["SYN-000001"].nature_of_business == "second"
    assert fold(events[::-1])["SYN-000001"].nature_of_business == "second"


def load_feed() -> tuple[list[tuple[str, Event]], Counter]:
    events, rejected, seen = [], Counter(), set()
    for line in FEED.read_text(encoding="utf-8").splitlines():
        raw = line.encode()
        mid = hashlib.sha256(raw).hexdigest()
        if mid in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(mid)
        try:
            events.append((mid, parse_message(raw)))
        except QuarantineError as exc:
            rejected[exc.reason] += 1
    return events, rejected


def test_feed_accounting():
    events, rejected = load_feed()
    final = fold(events)
    quarantined = sum(v for k, v in rejected.items() if k != "duplicate")
    assert len(events) + rejected["duplicate"] + quarantined == 2518
    assert rejected["duplicate"] == 68
    assert quarantined == 32
    assert len(final) == 2000  # no company lost
    assert sum(s.status == "struck_off" for s in final.values()) == 8


def test_feed_fold_is_order_independent():
    events, _ = load_feed()
    expected = fold(events)
    for seed in range(5):
        shuffled = events[:]
        random.Random(seed).shuffle(shuffled)
        assert fold(shuffled) == expected


def test_naive_timestamp_reading_matters_in_the_feed():
    """SYN-000521's naive 09:06:19 must be read as +08:00, else it would beat the real 16:27:19+08:00 update."""
    events, _ = load_feed()
    state = fold(events)["SYN-000521"]
    assert state.updated_at == datetime(2026, 8, 14, 8, 27, 19, tzinfo=UTC)
