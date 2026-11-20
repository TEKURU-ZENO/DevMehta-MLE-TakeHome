"""API contract: one test per status code the portal can see."""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from similar.api import create_app


def publish(processor, reg, description, *, type_="company-upserted", at="2026-08-14T09:00:00+08:00",
            industry="Food & Beverage"):
    payload = {"type": type_, "registrationNumber": reg, "updatedAt": at}
    if type_ == "company-upserted":
        payload["data"] = {"companyName": f"Company {reg}", "natureOfBusiness": description, "state": "Selangor",
                           "industry": industry}
    body = json.dumps(payload).encode()
    processor.handle(body, hashlib.sha256(body).hexdigest())


@pytest.fixture
def client(processor, database_url):
    publish(processor, "SYN-000001", "Operating cafes and coffee shops in Johor Bahru")
    publish(processor, "SYN-000002", "Operating cafes and coffee shops in Penang")
    publish(processor, "SYN-000003", "Operating cafes and coffee shops in Melaka")
    publish(processor, "SYN-000004", "Freight forwarding at Port Klang", industry="Logistics")
    publish(processor, "SYN-000005", "")  # no description
    publish(processor, "SYN-000006", "Operating cafes and coffee shops in Kuching")
    publish(processor, "SYN-000006", "", type_="company-struck-off", at="2026-08-14T10:00:00+08:00")
    with TestClient(create_app(database_url)) as c:
        yield c


def test_success_returns_cards_most_similar_first(client):
    r = client.get("/companies/SYN-000001/similar")
    assert r.status_code == 200
    body = r.json()
    assert body["company"]["registrationNumber"] == "SYN-000001"
    regs = [c["registrationNumber"] for c in body["similar"]]
    assert "SYN-000001" not in regs  # never itself
    assert "SYN-000005" not in regs  # no description, so not recommendable
    assert "SYN-000006" not in regs  # struck off
    assert set(regs[:2]) == {"SYN-000002", "SYN-000003"}
    assert regs[-1] == "SYN-000004"
    scores = [c["score"] for c in body["similar"]]
    assert scores == sorted(scores, reverse=True)
    assert set(body["similar"][0]) == {"registrationNumber", "companyName", "industry", "state", "score"}


def test_id_is_normalised_like_the_consumer(client):
    assert client.get("/companies/syn-000001/similar").status_code == 200


def test_limit(client):
    assert len(client.get("/companies/SYN-000001/similar?limit=1").json()["similar"]) == 1


@pytest.mark.parametrize("limit", ["0", "21", "abc"])
def test_bad_limit(client, limit):
    r = client.get(f"/companies/SYN-000001/similar?limit={limit}")
    assert r.status_code == 400
    assert r.json()["error"]["code"] in {"invalid_limit", "invalid_request"}


def test_invalid_id(client):
    r = client.get("/companies/not-a-company/similar")
    assert r.status_code == 400
    assert r.json() == {"error": {"code": "invalid_registration_number", "message": r.json()["error"]["message"]}}


def test_unknown_company(client):
    r = client.get("/companies/SYN-999999/similar")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "company_not_found"


def test_struck_off_company(client):
    r = client.get("/companies/SYN-000006/similar")
    assert r.status_code == 410
    assert r.json()["error"]["code"] == "company_struck_off"


def test_company_without_description(client):
    r = client.get("/companies/SYN-000005/similar")
    assert r.status_code == 200
    assert r.json()["similar"] == []
    assert r.json()["reason"] == "no_business_description"


def test_database_unavailable():
    app = create_app("postgresql://postgres:postgres@127.0.0.1:1/companies?connect_timeout=1")
    with TestClient(app) as c:
        r = c.get("/companies/SYN-000001/similar")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "database_unavailable"
