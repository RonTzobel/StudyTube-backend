FROM python:3.12-slim

# PYTHONUNBUFFERED=1  — send stdout/stderr directly to the terminal without
#                       buffering so `docker logs -f` shows output in real-time.
# PYTHONDONTWRITEBYTECODE=1 — do not write .pyc files inside the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies:
#   ffmpeg — required by the worker for audio extraction (ffmpeg -i video.mp4 ...)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source so Docker layer cache is
# preserved on source-only changes (requirements.txt rarely changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the uploads directory exists inside the image so the bind mount
# from docker-compose lands cleanly even on first run.
RUN mkdir -p /app/uploads

# Default command: FastAPI backend.
# The worker service overrides this with "python worker.py" in docker-compose.yml.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
