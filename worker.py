"""
worker.py — RQ worker entrypoint.

Start this process separately from the FastAPI backend:

    python worker.py

The worker connects to Redis, listens on the "default" queue, and executes
jobs enqueued by the backend (e.g. process_video_pipeline).

Redis URL is read from settings (REDIS_URL in .env).

Startup includes retry logic so the worker survives Redis being briefly
unavailable during Docker container startup races.
"""

import logging
import time

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from rq import Queue, Worker

from app.config.settings import settings

# Pre-import the jobs module so it is already in sys.modules when RQ resolves
# the dotted-path string "app.worker.jobs.process_video_pipeline".
# Without this, RQ's importlib.import_module("app.worker.jobs") call can fail
# with AttributeError when `app` is a namespace package (missing __init__.py).
# This also loads the Whisper model at startup — fail fast if unavailable.
import app.worker.jobs  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis connection retry configuration
# ---------------------------------------------------------------------------

_REDIS_RETRY_ATTEMPTS = 12   # total attempts before giving up
_REDIS_RETRY_DELAY_S  = 5    # seconds between attempts


def _connect_to_redis() -> Redis:
    """
    Open a Redis connection and verify it with PING.

    Retries up to _REDIS_RETRY_ATTEMPTS times with a fixed delay between
    attempts. This makes the worker tolerant of Redis starting a few seconds
    after the worker container — a common Docker Compose startup race.

    Raises:
        RedisConnectionError: if all retry attempts are exhausted.
    """
    conn = Redis.from_url(settings.REDIS_URL)

    for attempt in range(1, _REDIS_RETRY_ATTEMPTS + 1):
        try:
            conn.ping()
            _log.info(
                "worker | connected to Redis | url=%s | attempt=%d",
                settings.REDIS_URL,
                attempt,
            )
            return conn
        except RedisConnectionError as exc:
            if attempt == _REDIS_RETRY_ATTEMPTS:
                _log.error(
                    "worker | could not connect to Redis after %d attempts — giving up | "
                    "url=%s | last_error=%s",
                    _REDIS_RETRY_ATTEMPTS,
                    settings.REDIS_URL,
                    exc,
                )
                raise
            _log.warning(
                "worker | Redis not ready | attempt=%d/%d | retrying in %ds | error=%s",
                attempt,
                _REDIS_RETRY_ATTEMPTS,
                _REDIS_RETRY_DELAY_S,
                exc,
            )
            time.sleep(_REDIS_RETRY_DELAY_S)


if __name__ == "__main__":
    _log.info("worker | starting up | Redis URL: %s", settings.REDIS_URL)

    conn = _connect_to_redis()

    queues = [Queue("default", connection=conn)]
    _log.info(
        "worker | listening on queues: %s",
        [q.name for q in queues],
    )
    _log.info("worker | ready | waiting for jobs")

    worker = Worker(queues, connection=conn)
    worker.work(with_scheduler=False)
