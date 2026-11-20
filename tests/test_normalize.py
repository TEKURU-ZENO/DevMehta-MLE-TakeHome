"""Unit tests for the field normalisers: one rule per test, no database needed."""

import json
from datetime import UTC, datetime

import pytest

from similar.normalize import (
    EventType,
    QuarantineError,
    Reason,
    collapse_repetition,
    normalize_description,
    normalize_registration_number,
    normalize_state,
    normalize_updated_at,
    parse_message,
)

NOW = datetime(2026, 11, 19, tzinfo=UTC)


def body(**overrides) -> bytes:
    message = {
        "type": "company-upserted",
        "registrationNumber": "SYN-001550",
        "updatedAt": "2026-08-14T09:00:00+08:00",
        "data": {"companyName": "Sinar Teguh Bakery Sdn. Bhd.", "natureOfBusiness": "Operating cafes",
                 "state": "Selangor", "industry": "Food & Beverage"},
    }
    message.update(overrides)
    return json.dumps({k: v for k, v in message.items() if v is not ...}).encode()


class TestRegistrationNumber:
    @pytest.mark.parametrize("raw", ["SYN-001550", " SYN-001550 ", "syn-001550", "\tSyn-001550\n"])
    def test_variants_normalise_to_one_id(self, raw):
        assert normalize_registration_number(raw) == "SYN-001550"

    def test_missing(self):
        with pytest.raises(QuarantineError) as exc:
            normalize_registration_number(None)
        assert exc.value.reason is Reason.MISSING_REGISTRATION_NUMBER

    @pytest.mark.parametrize("raw", [64, 119, "SYN-1550", "SYN-0015500", "ABC-001550", "", "SYN 001550"])
    def test_invalid_is_rejected_not_guessed(self, raw):
        with pytest.raises(QuarantineError) as exc:
            normalize_registration_number(raw)
        assert exc.value.reason is Reason.INVALID_REGISTRATION_NUMBER


class TestUpdatedAt:
    def test_offset_converted_to_utc(self):
        assert normalize_updated_at("2026-08-14T09:00:00+08:00", now=NOW) == datetime(2026, 8, 14, 1, tzinfo=UTC)

    def test_z_suffix(self):
        assert normalize_updated_at("2026-08-14T01:00:00Z", now=NOW) == datetime(2026, 8, 14, 1, tzinfo=UTC)

    def test_naive_string_is_malaysia_time(self):
        assert normalize_updated_at("2026-08-14T09:00:00", now=NOW) == datetime(2026, 8, 14, 1, tzinfo=UTC)

    def test_unix_milliseconds(self):
        assert normalize_updated_at(1786670837000, now=NOW) == datetime(2026, 8, 14, 1, 27, 17, tzinfo=UTC)

    def test_unix_seconds(self):
        assert normalize_updated_at(1786670837, now=NOW) == datetime(2026, 8, 14, 1, 27, 17, tzinfo=UTC)

    def test_unix_digit_string(self):
        assert normalize_updated_at("1786670837000", now=NOW) == datetime(2026, 8, 14, 1, 27, 17, tzinfo=UTC)

    def test_missing(self):
        with pytest.raises(QuarantineError) as exc:
            normalize_updated_at(None, now=NOW)
        assert exc.value.reason is Reason.MISSING_UPDATED_AT

    @pytest.mark.parametrize("raw", ["yesterday", "2026-13-01T00:00:00", True, [], {}])
    def test_unparseable(self, raw):
        with pytest.raises(QuarantineError) as exc:
            normalize_updated_at(raw, now=NOW)
        assert exc.value.reason is Reason.INVALID_UPDATED_AT

    @pytest.mark.parametrize("raw", ["1970-01-02T00:00:00Z", "2099-01-01T00:00:00Z", 0])
    def test_implausible(self, raw):
        with pytest.raises(QuarantineError) as exc:
            normalize_updated_at(raw, now=NOW)
        assert exc.value.reason is Reason.IMPLAUSIBLE_UPDATED_AT


class TestCardFields:
    @pytest.mark.parametrize("raw", ["SELANGOR", "selangor", "  Selangor ", "SeLaNgOr"])
    def test_state_case(self, raw):
        assert normalize_state(raw) == "Selangor"

    def test_state_alias(self):
        assert normalize_state("Penang") == "Pulau Pinang"

    def test_unknown_state_kept(self):
        assert normalize_state("Atlantis") == "Atlantis"

    @pytest.mark.parametrize("raw", ["", "   ", None, 42])
    def test_empty_description_is_none(self, raw):
        assert normalize_description(raw) is None

    def test_repeated_description_collapsed(self):
        assert collapse_repetition(" ".join(["Hotel butik di Seremban"] * 60)) == "Hotel butik di Seremban"

    def test_genuine_repetition_kept(self):
        assert collapse_repetition("Bakery bakery cafe") == "Bakery bakery cafe"


class TestParseMessage:
    def test_clean_upsert(self):
        event = parse_message(body(), now=NOW)
        assert event.event_type is EventType.UPSERTED
        assert event.registration_number == "SYN-001550"
        assert event.company_name == "Sinar Teguh Bakery Sdn. Bhd."
        assert event.text_hash is not None

    def test_struck_off_needs_no_data(self):
        event = parse_message(body(type="company-struck-off", data=...), now=NOW)
        assert event.event_type is EventType.STRUCK_OFF

    def test_unknown_keys_kept_as_extra(self):
        event = parse_message(body(source="partner-api", ingestedBy="etl-v2"), now=NOW)
        assert event.extra == {"source": "partner-api", "ingestedBy": "etl-v2"}

    def test_empty_strings_become_null(self):
        event = parse_message(body(data={"companyName": "X", "natureOfBusiness": "", "state": None}), now=NOW)
        assert event.nature_of_business is None and event.state is None and event.industry is None
        assert event.text_hash is None

    @pytest.mark.parametrize(
        ("raw", "reason"),
        [
            (b'{"type": "company-upserted", "registrationNumber": "SYN-0', Reason.INVALID_JSON),
            (b"null", Reason.NOT_AN_OBJECT),
            (b"\xff\xfe", Reason.INVALID_ENCODING),
            (body(type="company-merged"), Reason.UNKNOWN_EVENT_TYPE),
            (body(registrationNumber=...), Reason.MISSING_REGISTRATION_NUMBER),
            (body(registrationNumber=119), Reason.INVALID_REGISTRATION_NUMBER),
            (body(updatedAt=...), Reason.MISSING_UPDATED_AT),
            (body(data=...), Reason.MISSING_DATA),
            (body(data="companyName=Foo; state=Perak"), Reason.INVALID_DATA),
        ],
    )
    def test_permanent_failures(self, raw, reason):
        with pytest.raises(QuarantineError) as exc:
            parse_message(raw, now=NOW)
        assert exc.value.reason is reason
