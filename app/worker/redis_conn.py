"""
redis_conn.py — Shared Redis connection and RQ queue.

Imported by:
  - app/routers/videos.py  (to enqueue jobs)
  - worker.py              (to start the RQ worker)

The connection is lazy — redis-py does not actually open a socket until the
first command is issued, so importing this module in the FastAPI process is
safe even if Redis is temporarily unavailable at startup.
"""

from redis import Redis
from rq import Queue

from app.config.settings import settings

# Single Redis connection shared within this process.
redis_conn = Redis.from_url(settings.REDIS_URL)

# Default queue — all video pipeline jobs go here.
default_queue = Queue(connection=redis_conn)
