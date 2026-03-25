"""
worker.py — RQ worker entrypoint.

Start this process separately from the FastAPI backend:

    python worker.py

The worker connects to Redis, listens on the "default" queue, and executes
jobs enqueued by the backend (e.g. process_video_pipeline).

Redis URL is read from settings (REDIS_URL in .env).
"""

import logging

from redis import Redis
from rq import Queue, Worker

from app.config.settings import settings

# Pre-import the jobs module so it is already in sys.modules when RQ resolves
# the dotted-path string "app.worker.jobs.process_video_pipeline".
# Without this, RQ's importlib.import_module("app.worker.jobs") call can fail
# with AttributeError when `app` is a namespace package (missing __init__.py).
# This also loads the Whisper model at startup, which is the intended behaviour
# for the worker process (fail fast if the model is unavailable).
import app.worker.jobs  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

if __name__ == "__main__":
    conn = Redis.from_url(settings.REDIS_URL)
    queues = [Queue("default", connection=conn)]
    worker = Worker(queues, connection=conn)
    worker.work(with_scheduler=False)
