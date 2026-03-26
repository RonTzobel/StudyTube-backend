# StudyTube

An AI-powered lecture learning platform. Upload a lecture video — get a searchable transcript, instant answers, and auto-generated quizzes.

**Live:** [www.studytubeapp.com](https://www.studytubeapp.com)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Architecture Overview](#architecture-overview)
- [AI Processing Pipeline](#ai-processing-pipeline)
- [Backend Design](#backend-design)
- [Frontend Design](#frontend-design)
- [Infrastructure & Deployment](#infrastructure--deployment)
- [CI/CD Flow](#cicd-flow)
- [Engineering Challenges Solved](#engineering-challenges-solved)
- [Scalability Considerations](#scalability-considerations)
- [Example User Flow](#example-user-flow)
- [Future Improvements](#future-improvements)
- [Tech Stack Summary](#tech-stack-summary)
- [Author](#author)

---

## Project Overview

StudyTube converts lecture videos into structured, queryable knowledge. A user uploads an MP4 lecture; the system transcribes it, segments the transcript into semantic chunks, embeds each chunk, and stores the vectors in PostgreSQL. From that point the user can:

- **Search** the lecture semantically — not just keyword matching
- **Ask questions** answered via Retrieval-Augmented Generation (RAG)
- **Generate quizzes** drawn from the actual lecture content
- **Read the full transcript** with tutor-style follow-up support

The backend is a production FastAPI service deployed on AWS EC2 behind a custom domain, with a Redis-backed worker queue, Docker Compose orchestration, and an automated GitHub Actions CI/CD pipeline.

---

## Live Demo

[https://www.studytubeapp.com](https://www.studytubeapp.com)

Register, upload a lecture video, and within a few minutes the transcript, search, Q&A, and quiz features become available. Processing status is tracked in real time via polling the video status endpoint.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          AWS EC2 Instance                            │
│                                                                      │
│  ┌─────────────┐    ┌────────────────┐    ┌────────────────────┐    │
│  │   React     │    │    FastAPI     │    │   PostgreSQL       │    │
│  │   (Vite)    │◄──►│   Backend     │◄──►│   (Aiven Cloud)    │    │
│  │   Nginx     │    │   :8000        │    │                    │    │
│  └─────────────┘    └───────┬────────┘    └────────────────────┘    │
│                             │                                        │
│                       enqueue job                                    │
│                             │                                        │
│                      ┌──────▼───────┐    ┌────────────────────┐    │
│                      │    Redis     │    │    RQ Worker        │    │
│                      │  (queue)     │◄──►│  (video pipeline)   │    │
│                      └──────────────┘    └──────────┬─────────┘    │
│                                                      │              │
│                                              ┌───────▼──────┐       │
│                                              │    AWS S3    │       │
│                                              │  (video      │       │
│                                              │   storage)   │       │
│                                              └──────────────┘       │
└──────────────────────────────────────────────────────────────────────┘

External APIs
  OpenAI Whisper API  ── audio transcription (no local model inference)
  OpenAI Chat API     ── RAG answers, quiz generation, summaries
```

The backend and worker run as **separate Docker containers** sharing a PostgreSQL database. The FastAPI process never imports worker code — the worker is a completely separate process that consumes jobs from the Redis queue. This keeps the API server lightweight and ensures the ML pipeline runs in isolation.

---

## AI Processing Pipeline

After a video is uploaded to S3, the backend enqueues a job. The worker processes it through five sequential stages, each committed to the `videos.status` column in PostgreSQL so the frontend can display live progress:

```
uploaded
   │
   ▼  POST /api/v1/videos/{id}/transcribe
queued ──────────────────────────────────── job pushed to Redis
   │
   ▼  worker picks up job
processing ──► ffmpeg extracts mono 16 kHz WAV from S3 video
   │
   ▼
transcribing ──► WAV split into 5-minute chunks (ffmpeg -f segment)
              ──► each chunk uploaded to OpenAI Whisper API
              ──► chunk transcripts merged into full text
              ──► transcript row saved to PostgreSQL
   │
   ▼
embedding ──► sentence-aware overlapping chunking
           ──►   target: ~800 words per chunk
           ──►   overlap: ~120 words (preserves cross-boundary context)
           ──► each chunk embedded via BAAI/bge-small-en-v1.5 (384-dim)
           ──► vectors stored in PostgreSQL
   │
   ▼
completed
   │
   └──► (on any exception) → failed  [error_message column set]
```

**Chunking strategy:** Chunks are built sentence-by-sentence to a target word count with a configurable overlap. Sentence boundaries are respected to avoid splitting mid-thought — important for RAG retrieval quality where a half-sentence chunk produces poor answers.

**Embedding model:** `BAAI/bge-small-en-v1.5` — a top-performing English retrieval model at 384 dimensions, efficient on CPU. Critically, the **same model instance** is used at index time (worker, embedding chunks) and query time (backend, embedding search queries), ensuring geometric compatibility of all vectors.

**RAG quality gate:** A two-level similarity threshold prevents low-quality retrievals from reaching the LLM:

| Cosine similarity | Action |
|---|---|
| Below `RAG_LOW_THRESHOLD` (0.30) | Immediate fallback — no OpenAI call |
| Between thresholds | Borderline — OpenAI called with wider chunk window, `confidence_level: "low"` |
| Above `RAG_GOOD_THRESHOLD` (0.60) | High-confidence answer, `confidence_level: "high"` |

---

## Backend Design

```
app/
├── config/
│   └── settings.py               # pydantic-settings; all config via env vars
├── core/
│   ├── dependencies.py           # FastAPI DI (get_current_user, get_session)
│   └── security.py               # bcrypt hashing, JWT creation/verification
├── database/
│   └── session.py                # SQLAlchemy engine + session factory
├── models/                       # SQLModel table definitions (DB schema)
│   ├── user.py
│   ├── video.py                  # includes status + error_message fields
│   ├── transcript.py
│   └── chunk.py
├── schemas/                      # Pydantic request/response contracts
│   ├── auth.py                   # RegisterRequest, LoginRequest, UserPublic
│   ├── video.py                  # VideoRead, TranscribeAccepted, UploadResponse
│   └── qa.py                     # AskRequest, AskResponse, QuizResponse
├── routers/                      # Thin HTTP layer — delegates to services
│   ├── auth.py                   # register, login, /me, Google OAuth
│   └── videos.py                 # upload, transcribe, search, ask, quiz, delete
├── services/                     # All business logic
│   ├── user_service.py
│   ├── video_service.py
│   ├── transcript_service.py     # transcript DB helpers
│   ├── transcription_service.py  # OpenAI Whisper API integration
│   ├── chunk_service.py          # sentence-aware overlapping chunking
│   ├── embedding_service.py      # sentence-transformers model + encode()
│   ├── retrieval_service.py      # cosine similarity search over stored vectors
│   ├── qa_service.py             # RAG pipeline with two-level quality gate
│   ├── quiz_service.py           # quiz generation with even chunk sampling
│   └── s3_service.py             # S3 upload/download
└── worker/
    ├── redis_conn.py             # Redis connection + default queue object
    └── jobs.py                   # pipeline job: extraction → transcription → embedding
```

**Design principles:**

- **Routers are thin:** every router function validates HTTP input, calls one service, returns a schema. No business logic in routers.
- **Models vs schemas:** SQLModel table definitions are never exposed directly in API responses. Pydantic schemas define the API contract independently.
- **Worker isolation:** `app.worker.jobs` is never imported by the FastAPI process. Sentence-transformers and Whisper-related code loads only in the worker container.
- **Fail-safe status tracking:** every pipeline stage commits its status to PostgreSQL immediately. A crash at any stage leaves the video in a diagnosable state with `error_message` set.
- **Idempotent pipeline stages:** the chunking stage deletes existing chunks before inserting new ones, making the entire pipeline safe to re-run on retry.

**Authentication:**
- Email/password: bcrypt hashing + JWT (HS256, `python-jose`)
- Google OAuth2: server-side authorization code flow with `httpx` token exchange
- Pydantic `@field_validator` enforces bcrypt's 72 UTF-8 byte password limit at the schema layer — a naive `len()` check would pass a 37-character Hebrew password that encodes to 74 bytes

---

## Frontend Design

Built with **React + Vite**, served via Nginx on the same EC2 instance.

- **Auth flow:** JWT stored in localStorage; an Axios request interceptor attaches `Authorization: Bearer` on every request automatically
- **Google OAuth:** redirect-based flow; the backend exchanges the OAuth code server-side, issues an app JWT, and redirects to `/auth/callback?token=` for seamless client-side storage — the same storage path as email/password login
- **Live status polling:** after triggering transcription, the frontend polls `GET /api/v1/videos/{id}` until status becomes `completed` or `failed`, displaying the current stage to the user
- **Semantic search:** debounced query input sends `POST /search` and renders ranked results with similarity scores and source chunk text

---

## Infrastructure & Deployment

**Docker Compose** orchestrates three services on a single EC2 instance:

```yaml
services:
  backend:  FastAPI (uvicorn)       restart: unless-stopped
  worker:   RQ worker (jobs.py)     restart: unless-stopped
  redis:    Redis 7 (queue broker)  restart: unless-stopped
```

- **Database:** Aiven Cloud PostgreSQL — managed, off-instance, automatic backups
- **Object storage:** AWS S3 — videos are stored on S3; EC2 holds no persistent media
- **Schema migrations:** Alembic — the deploy pipeline runs `alembic upgrade head` inside a one-shot container before bringing the stack up, ensuring migrations land before the app starts accepting traffic
- **TLS:** Nginx on the host proxies requests to the FastAPI backend and serves the React bundle; Let's Encrypt certificates handle TLS termination

---

## CI/CD Flow

```
git push → main
    │
    ▼
GitHub Actions  (.github/workflows/deploy.yml)
    │
    ├── SSH into EC2
    ├── git pull
    ├── docker compose build
    ├── docker compose run --rm --no-deps backend alembic upgrade head
    └── docker compose up -d
```

Images are built before the old containers are stopped, minimising downtime. The Alembic step runs as a one-shot container against the live database, so schema changes are applied before any new application code starts serving requests.

---

## Engineering Challenges Solved

**OOM crashes during long-video transcription**
Running `faster-whisper` locally on a CPU-only EC2 instance caused the worker process to be killed by the Linux OOM killer (signal 9) on videos over 30 minutes. Even after reducing the model from `medium.en` → `small.en` → `base.en` and implementing chunked inference, the CTranslate2 runtime deadlocked under Docker's CPU scheduling — process alive, CPU near zero, no progress. The solution was to replace all local model inference with the OpenAI Whisper API. Audio is split into 5-minute chunks with ffmpeg before upload to stay under the 25 MB API limit. Local RAM pressure is eliminated entirely.

**Worker startup race condition**
Docker's `depends_on: service_healthy` does not guarantee application-level readiness at the exact moment the dependent container starts. The RQ worker was crashing on launch if Redis was still initialising. Fixed with an explicit retry loop (12 attempts, 5-second intervals) that pings Redis before starting the worker — making container startup order-independent.

**FastAPI import in worker image**
`app/services/s3_service.py` had `from fastapi import UploadFile` at module level. The worker imports `s3_service` to call `download_file_from_s3`, which triggered the FastAPI import — but FastAPI is not installed in the worker image. Fixed by converting the type annotation to a forward-reference string (`"UploadFile"`). Since `UploadFile` is only used as a type hint and never evaluated at runtime, no import is needed at all.

**passlib/bcrypt Docker version mismatch**
`passlib==1.7.4` reads `bcrypt.__about__` at initialisation, a module removed in `bcrypt>=4.1`. Local development used `bcrypt==3.2.2` and worked fine; Docker resolved the latest (`4.2.x`) and failed at the first password hash with `AttributeError`. Fixed by pinning `bcrypt==4.0.1` in `requirements.backend.txt` with a detailed comment explaining the compatibility boundary.

**RQ job path resolution failure**
The `app/` directory was missing `__init__.py`, making it a Python namespace package. RQ resolves jobs by dotted-path string using `importlib.import_module` followed by `getattr` traversal — which silently fails for namespace packages because submodules are not attributes. Fixed by adding `app/__init__.py` and pre-importing `app.worker.jobs` in `worker.py` so the module is in `sys.modules` before RQ's resolver runs.

**bcrypt 72 UTF-8 byte limit**
bcrypt silently truncates passwords at 72 bytes. A 37-character Hebrew password is only 37 characters long but 74 UTF-8 bytes — it passes `len(password) <= 72` but is silently weakened at the hash boundary. Fixed with a Pydantic `@field_validator` that checks `len(password.encode("utf-8")) > 72`, returning a user-friendly error message without exposing the implementation detail.

---

## Scalability Considerations

| Concern | Current | Path to scale |
|---|---|---|
| Video storage | AWS S3 | Already horizontally unlimited |
| Database | Aiven managed PostgreSQL | Read replicas; pgBouncer connection pooling |
| Transcription | OpenAI Whisper API | Stateless API — parallelism is free |
| Embedding search | Cosine similarity in PostgreSQL | pgvector + HNSW index for large corpora |
| Worker throughput | Single RQ worker | Multiple workers; priority queues; horizontal scaling |
| Job durability | Redis (in-memory) | Celery + SQS for persistent job store |

The current single-instance architecture is intentional for an MVP. Service boundaries — API, worker, database, object storage all separated from day one — mean each layer scales independently without architectural changes.

---

## Example User Flow

```
1. Register
   POST /api/v1/auth/register
   ← { access_token, token_type, user: { id, email, username } }

2. Upload video
   POST /api/v1/videos/upload   (multipart/form-data, MP4)
   ← { video_id, s3_key, status: "queued" }

3. Start processing
   POST /api/v1/videos/{id}/transcribe
   ← 202 { message, video_id, status: "queued" }

   [worker pipeline runs in background]
   status:  queued → processing → transcribing → embedding → completed

4. Poll for completion
   GET /api/v1/videos/{id}
   ← { id, title, status: "completed", error_message: null, ... }

5. Semantic search
   POST /api/v1/videos/{id}/search
   { "query": "what is gradient descent?" }
   ← [{ chunk_text, similarity_score }, ...]

6. Ask a question (RAG)
   POST /api/v1/videos/{id}/ask
   { "question": "Explain backpropagation in simple terms" }
   ← { answer, retrieved_chunks, confidence_level, grounded: true }

7. Generate quiz
   POST /api/v1/videos/{id}/quiz
   { "num_questions": 5 }
   ← { questions: [{ question, options, correct_answer }, ...] }
```

---

## Future Improvements

- **pgvector:** Replace manual cosine similarity with PostgreSQL's `pgvector` extension and an HNSW index for sub-millisecond retrieval at scale
- **Timestamp-aware chunks:** Attach video timestamps to each chunk so search results link directly to the relevant moment in the video player
- **Speaker diarisation:** Identify and label different speakers in lecture recordings
- **Streaming answers:** Stream OpenAI Chat responses token-by-token via Server-Sent Events for faster perceived response time
- **Celery + SQS:** Replace RQ with a more durable queue backend for production-grade job persistence and retry semantics
- **Rate limiting:** Per-user upload and API call rate limits enforced at the Nginx or FastAPI middleware layer
- **Forgot password flow:** Currently Google-only users cannot set a password for email/password login; a token-based password-set flow is the natural next step

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| ORM | SQLModel (SQLAlchemy + Pydantic) |
| Database | PostgreSQL (Aiven Cloud) |
| Schema migrations | Alembic |
| Auth | bcrypt + JWT + Google OAuth2 |
| Transcription | OpenAI Whisper API |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| LLM | OpenAI Chat API |
| Job queue | Redis + RQ |
| Audio processing | ffmpeg |
| Object storage | AWS S3 + boto3 |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Reverse proxy | Nginx + Let's Encrypt |
| Cloud | AWS EC2 |
| Frontend | React + Vite |

---

## Author

Built and deployed by Ronzo.

[www.studytubeapp.com](https://www.studytubeapp.com)
