"""Ack/nack decisions of the consumer, with a fake message and a fake processor (no RabbitMQ, no database)."""

import asyncio

import psycopg
import pytest

from similar import consumer as consumer_module
from similar.config import Settings
from similar.consumer import MAX_ATTEMPTS, Consumer
from similar.db import Outcome
from similar.normalize import Reason

SETTINGS = Settings("", "", "q", "hash-bow-384", 384, prefetch=4, workers=2)


class FakeMessage:
    def __init__(self, body=b"{}", message_id="m1"):
        self.body, self.message_id = body, message_id
        self.acks, self.nacks = 0, []

    async def ack(self):
        self.acks += 1

    async def nack(self, requeue=True):
        self.nacks.append(requeue)


class FakeProcessor:
    def __init__(self, error=None):
        self.error = error
        self.quarantined = []
        self.embedder = type("E", (), {"name": "fake"})()

    def handle(self, body, message_id):
        if self.error:
            raise self.error
        return Outcome.APPLIED

    def quarantine(self, body, message_id, reason, detail):
        self.quarantined.append((message_id, reason))
        return Outcome.QUARANTINED


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr(consumer_module.asyncio, "sleep", instant)


def deliver(processor, message, times=1):
    async def run():
        c = Consumer(processor, SETTINGS)
        for _ in range(times):
            await c.on_message(message)
        return c

    return asyncio.run(run())


def test_success_is_acked_once():
    msg = FakeMessage()
    deliver(FakeProcessor(), msg)
    assert msg.acks == 1 and msg.nacks == []


def test_database_down_is_requeued_never_quarantined():
    msg = FakeMessage()
    processor = FakeProcessor(psycopg.OperationalError("connection refused"))
    deliver(processor, msg, times=MAX_ATTEMPTS + 3)
    assert msg.acks == 0
    assert msg.nacks == [True] * (MAX_ATTEMPTS + 3)
    assert processor.quarantined == []


def test_poison_message_is_quarantined_after_max_attempts():
    msg = FakeMessage()
    processor = FakeProcessor(RuntimeError("bug"))
    deliver(processor, msg, times=MAX_ATTEMPTS)
    assert msg.nacks == [True] * (MAX_ATTEMPTS - 1)
    assert msg.acks == 1
    assert processor.quarantined == [("m1", Reason.PROCESSING_ERROR)]
