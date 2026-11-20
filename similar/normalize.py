"""Turning one raw feed message into a clean event, or a reason it can never be applied.

Every rule about the feed's quirks lives here, as small pure functions, so the consumer, the API and the tests all
agree on what a registration number or a timestamp means. See the assumptions table in DESIGN.md.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

MALAYSIA_TZ = timezone(timedelta(hours=8), "MYT")
REGISTRATION_RE = re.compile(r"SYN-\d{6}")
# Unix times at or above this are milliseconds. 1e11 seconds is the year 5138; 1e11 ms is 1973.
MILLISECONDS_THRESHOLD = 100_000_000_000
EARLIEST_PLAUSIBLE = datetime(2000, 1, 1, tzinfo=UTC)
# An event dated far in the future would win against every real update forever, so we refuse it.
FUTURE_TOLERANCE = timedelta(days=1)
MAX_DESCRIPTION_CHARS = 1000

KNOWN_TOP_LEVEL_KEYS = frozenset({"type", "registrationNumber", "updatedAt", "data"})
CARD_FIELDS = {"companyName": "company_name", "natureOfBusiness": "nature_of_business", "state": "state",
               "industry": "industry"}

_STATES = ["Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang", "Perak", "Perlis", "Pulau Pinang",
           "Sabah", "Sarawak", "Selangor", "Terengganu", "Kuala Lumpur", "Labuan", "Putrajaya"]
_STATE_LOOKUP = {s.casefold(): s for s in _STATES} | {
    "penang": "Pulau Pinang", "malacca": "Melaka", "kl": "Kuala Lumpur", "wp kuala lumpur": "Kuala Lumpur",
}


class EventType(StrEnum):
    UPSERTED = "company-upserted"
    STRUCK_OFF = "company-struck-off"


class Reason(StrEnum):
    """Why a message was quarantined. Each one is permanent: retrying the same bytes can never succeed."""

    INVALID_ENCODING = "invalid_encoding"
    INVALID_JSON = "invalid_json"
    NOT_AN_OBJECT = "not_an_object"
    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    MISSING_REGISTRATION_NUMBER = "missing_registration_number"
    INVALID_REGISTRATION_NUMBER = "invalid_registration_number"
    MISSING_UPDATED_AT = "missing_updated_at"
    INVALID_UPDATED_AT = "invalid_updated_at"
    IMPLAUSIBLE_UPDATED_AT = "implausible_updated_at"
    MISSING_DATA = "missing_data"
    INVALID_DATA = "invalid_data"
    PROCESSING_ERROR = "processing_error"  # set by the consumer after repeated unexpected failures, not here


class QuarantineError(ValueError):
    def __init__(self, reason: Reason, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Event:
    registration_number: str
    event_type: EventType
    updated_at: datetime  # always UTC
    company_name: str | None = None
    nature_of_business: str | None = None
    state: str | None = None
    industry: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # unknown keys, kept for audit

    @property
    def text_hash(self) -> str | None:
        return description_hash(self.nature_of_business)


# --- field normalisers ---------------------------------------------------------------------------------------------


def normalize_registration_number(value: object) -> str:
    """'  syn-001550 ' -> 'SYN-001550'. Anything else that isn't SYN-dddddd is rejected, never guessed."""
    if value is None:
        raise QuarantineError(Reason.MISSING_REGISTRATION_NUMBER, "registrationNumber is missing")
    if not isinstance(value, str):
        # e.g. the bare integers 64 / 119 in the feed. Zero-padding them into SYN-000064 would be a guess.
        raise QuarantineError(Reason.INVALID_REGISTRATION_NUMBER, f"expected a string, got {type(value).__name__}")
    cleaned = value.strip().upper()
    if not REGISTRATION_RE.fullmatch(cleaned):
        raise QuarantineError(Reason.INVALID_REGISTRATION_NUMBER, f"{value!r} is not SYN-dddddd")
    return cleaned


def normalize_updated_at(value: object, *, now: datetime | None = None) -> datetime:
    """Return the event time in UTC.

    - ISO string with an offset or Z: converted to UTC.
    - ISO string without an offset: assumed Malaysia time (+08:00), like the rest of the feed.
    - Integer (or digit string): Unix time; milliseconds if >= 1e11, otherwise seconds.
    - Missing, unparseable, before 2000 or more than a day in the future: rejected.
    """
    if value is None:
        raise QuarantineError(Reason.MISSING_UPDATED_AT, "updatedAt is missing")
    if isinstance(value, bool):
        raise QuarantineError(Reason.INVALID_UPDATED_AT, "updatedAt is a boolean")
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value.strip())
    if isinstance(value, (int, float)):
        seconds = value / 1000 if abs(value) >= MILLISECONDS_THRESHOLD else value
        try:
            parsed = datetime.fromtimestamp(seconds, UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise QuarantineError(Reason.INVALID_UPDATED_AT, f"{value!r} is not a usable Unix time") from exc
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError as exc:
            raise QuarantineError(Reason.INVALID_UPDATED_AT, f"{value!r} is not ISO 8601") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=MALAYSIA_TZ)
        parsed = parsed.astimezone(UTC)
    else:
        raise QuarantineError(Reason.INVALID_UPDATED_AT, f"unsupported type {type(value).__name__}")

    now = now or datetime.now(UTC)
    if parsed < EARLIEST_PLAUSIBLE or parsed > now + FUTURE_TOLERANCE:
        raise QuarantineError(Reason.IMPLAUSIBLE_UPDATED_AT, f"{parsed.isoformat()} is outside the plausible range")
    return parsed


def clean_text(value: object) -> str | None:
    """Trim and collapse whitespace; empty strings and non-strings become None."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def collapse_repetition(text: str) -> str:
    """'A B A B A B' -> 'A B'. Some upstream sources concatenate the same description dozens of times."""
    words = text.split(" ")
    n = len(words)
    for period in range(1, n // 2 + 1):
        if n % period == 0 and words == words[:period] * (n // period):
            return " ".join(words[:period])
    return text


def normalize_description(value: object) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return collapse_repetition(text)[:MAX_DESCRIPTION_CHARS]


def normalize_state(value: object) -> str | None:
    """'SELANGOR' / 'selangor' -> 'Selangor'. Unrecognised values are kept as given (trimmed), not dropped."""
    text = clean_text(value)
    if text is None:
        return None
    return _STATE_LOOKUP.get(text.casefold(), text)


def description_hash(description: str | None) -> str | None:
    if description is None:
        return None
    return hashlib.sha256(description.encode()).hexdigest()


# --- whole message -------------------------------------------------------------------------------------------------


def parse_message(body: bytes, *, now: datetime | None = None) -> Event:
    """Parse and normalise one message body. Raises QuarantineError if it can never be applied."""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QuarantineError(Reason.INVALID_ENCODING, str(exc)) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QuarantineError(Reason.INVALID_JSON, str(exc)) from exc
    if not isinstance(payload, dict):
        raise QuarantineError(Reason.NOT_AN_OBJECT, f"top level is {type(payload).__name__}")

    try:
        event_type = EventType(payload.get("type"))
    except ValueError as exc:
        raise QuarantineError(Reason.UNKNOWN_EVENT_TYPE, f"type={payload.get('type')!r}") from exc

    registration_number = normalize_registration_number(payload.get("registrationNumber"))
    updated_at = normalize_updated_at(payload.get("updatedAt"), now=now)
    extra = {k: v for k, v in payload.items() if k not in KNOWN_TOP_LEVEL_KEYS}

    if event_type is EventType.STRUCK_OFF:
        return Event(registration_number, event_type, updated_at, extra=extra)

    data = payload.get("data")
    if data is None:
        raise QuarantineError(Reason.MISSING_DATA, "company-upserted without data")
    if not isinstance(data, dict):
        raise QuarantineError(Reason.INVALID_DATA, f"data is {type(data).__name__}, expected an object")
    unknown_data = {k: v for k, v in data.items() if k not in CARD_FIELDS}
    if unknown_data:
        extra["data"] = unknown_data
    return Event(
        registration_number=registration_number,
        event_type=event_type,
        updated_at=updated_at,
        company_name=clean_text(data.get("companyName")),
        nature_of_business=normalize_description(data.get("natureOfBusiness")),
        state=normalize_state(data.get("state")),
        industry=clean_text(data.get("industry")),
        extra=extra,
    )


def message_id_for(body: bytes, amqp_message_id: str | None) -> str:
    """The AMQP message_id, or the same sha256-of-body the producer uses if a message ever arrives without one."""
    return amqp_message_id or hashlib.sha256(body).hexdigest()
