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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

if __name__ == "__main__":
    conn = Redis.from_url(settings.REDIS_URL)
    queues = [Queue("default", connection=conn)]
    worker = Worker(queues, connection=conn)
    worker.work(with_scheduler=False)
