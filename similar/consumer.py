"""Reads company-events and applies each message to Postgres.

Acknowledgement rules (at-least-once in, exactly-once effect):
    applied / duplicate / quarantined  -> ack, after the transaction has committed
    database unavailable               -> wait (exponential backoff), then requeue. Never quarantined.
    anything unexpected                -> requeue up to MAX_ATTEMPTS, then quarantine as processing_error and ack,
                                          so one poison message can't block the queue and is still accounted for.

Usage: uv run python -m similar.consumer [--exit-when-idle SECONDS]
"""

import argparse
import asyncio
import collections
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import aio_pika
import psycopg
import psycopg_pool
from aio_pika.abc import AbstractIncomingMessage

from similar.config import Settings, load_settings
from similar.db import Outcome, Processor, create_pool, init_schema
from similar.embedding import build_embedder
from similar.normalize import Reason, message_id_for

log = logging.getLogger("similar.consumer")

MAX_ATTEMPTS = 5
TRANSIENT_ERRORS = (psycopg.OperationalError, psycopg_pool.PoolTimeout)


class Consumer:
    def __init__(self, processor: Processor, settings: Settings):
        self.processor = processor
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="apply")
        self.semaphore = asyncio.Semaphore(settings.workers)
        self.attempts: dict[str, int] = collections.defaultdict(int)
        self.transient_streak = 0
        self.counts: collections.Counter[str] = collections.Counter()
        self.last_activity = time.monotonic()
        self.in_flight = 0

    async def on_message(self, message: AbstractIncomingMessage) -> None:
        self.in_flight += 1
        try:
            async with self.semaphore:
                await self._process(message)
        finally:
            self.in_flight -= 1
            self.last_activity = time.monotonic()

    async def _process(self, message: AbstractIncomingMessage) -> None:
        loop = asyncio.get_running_loop()
        message_id = message_id_for(message.body, message.message_id)
        try:
            outcome = await loop.run_in_executor(self.executor, self.processor.handle, message.body, message_id)
        except TRANSIENT_ERRORS as exc:
            self.transient_streak += 1
            delay = min(30.0, 0.5 * 2 ** min(self.transient_streak, 6))
            log.warning("transient error, requeue in %.1fs: %s message_id=%s", delay, exc, message_id)
            self.counts["transient_retry"] += 1
            await asyncio.sleep(delay)
            await message.nack(requeue=True)
            return
        except Exception as exc:
            self.attempts[message_id] += 1
            attempt = self.attempts[message_id]
            if attempt < MAX_ATTEMPTS:
                log.exception("unexpected error (attempt %d/%d), requeue message_id=%s", attempt, MAX_ATTEMPTS,
                              message_id)
                self.counts["unexpected_retry"] += 1
                await asyncio.sleep(min(10.0, 0.5 * 2 ** attempt))
                await message.nack(requeue=True)
                return
            log.error("giving up after %d attempts, quarantining message_id=%s", attempt, message_id)
            try:
                outcome = await loop.run_in_executor(
                    self.executor, self.processor.quarantine, message.body, message_id, Reason.PROCESSING_ERROR,
                    f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                log.exception("could not quarantine, requeue message_id=%s", message_id)
                await message.nack(requeue=True)
                return
            self.attempts.pop(message_id, None)

        self.transient_streak = 0
        self.attempts.pop(message_id, None)
        await message.ack()
        self.counts[str(outcome)] += 1
        total = sum(self.counts[str(o)] for o in Outcome)
        if total % 500 == 0:
            log.info("progress %s", dict(self.counts))

    async def run(self, exit_when_idle: float | None) -> None:
        connection = await aio_pika.connect_robust(self.settings.rabbitmq_url)
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=self.settings.prefetch)
            # Same declaration as the registry team's (durable, no arguments), so this is a no-op if it exists.
            queue = await channel.declare_queue(self.settings.queue, durable=True)
            await queue.consume(self.on_message)
            log.info("consuming %s (prefetch=%d, workers=%d, model=%s)", self.settings.queue, self.settings.prefetch,
                     self.settings.workers, self.processor.embedder.name)
            while True:
                await asyncio.sleep(1)
                idle = time.monotonic() - self.last_activity
                if exit_when_idle is not None and idle >= exit_when_idle and self.in_flight == 0:
                    log.info("idle for %.0fs, exiting. totals %s", idle, dict(self.counts))
                    return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exit-when-idle", type=float, default=None,
                        help="stop after this many seconds without a message (handy for scripts and demos)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    settings = load_settings()
    init_schema(settings.database_url, settings.embedding_dim)
    embedder = build_embedder(settings.embedding_model, settings.embedding_dim)
    pool = create_pool(settings.database_url, max_size=settings.workers + 2)
    consumer = Consumer(Processor(pool, embedder), settings)
    try:
        asyncio.run(consumer.run(args.exit_when_idle))
    except KeyboardInterrupt:
        log.info("stopped. totals %s", dict(consumer.counts))
    finally:
        consumer.executor.shutdown(wait=True)
        pool.close()


if __name__ == "__main__":
    main()
