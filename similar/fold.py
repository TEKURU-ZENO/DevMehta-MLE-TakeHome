"""The "latest data wins" rule as a pure function.

The database applies the same rule in SQL (see db.REFRESH_COMPANY_SQL). This version is the reference the tests check
the database against, and what the offline evaluation uses to build the final company list from the feed file.

Order of events for one company, oldest to newest:
    1. updatedAt
    2. on an exact tie, a struck-off event beats an upsert (the safer outcome: don't recommend a dead company)
    3. then message_id, so the result never depends on arrival order

The company's status comes from the newest event. Its card (name, description, state, industry) comes from the newest
*upsert*, so a struck-off company keeps its last known details and an old upsert arriving after a tombstone can't
revive it.
"""

from dataclasses import dataclass
from datetime import datetime

from similar.normalize import Event, EventType


@dataclass(frozen=True)
class CompanyState:
    registration_number: str
    status: str  # "active" | "struck_off"
    updated_at: datetime
    last_message_id: str
    company_name: str | None
    nature_of_business: str | None
    state: str | None
    industry: str | None
    text_hash: str | None


def order_key(event: Event, message_id: str) -> tuple:
    return (event.updated_at, event.event_type is EventType.STRUCK_OFF, message_id)


def fold(events: list[tuple[str, Event]]) -> dict[str, CompanyState]:
    """Given (message_id, event) pairs in any order, return the final state per registration number."""
    by_company: dict[str, list[tuple[str, Event]]] = {}
    for message_id, event in events:
        by_company.setdefault(event.registration_number, []).append((message_id, event))

    result = {}
    for reg, items in by_company.items():
        items.sort(key=lambda item: order_key(item[1], item[0]))
        latest_id, latest = items[-1]
        upserts = [e for _, e in items if e.event_type is EventType.UPSERTED]
        card = upserts[-1] if upserts else None
        result[reg] = CompanyState(
            registration_number=reg,
            status="struck_off" if latest.event_type is EventType.STRUCK_OFF else "active",
            updated_at=latest.updated_at,
            last_message_id=latest_id,
            company_name=card.company_name if card else None,
            nature_of_business=card.nature_of_business if card else None,
            state=card.state if card else None,
            industry=card.industry if card else None,
            text_hash=card.text_hash if card else None,
        )
    return result
