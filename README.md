# StudyTube — Backend

AI-powered learning assistant for videos.
Built with **FastAPI** + **SQLModel** + **PostgreSQL**.

---

## Project Structure

```
backend/
├── app/
│   ├── main.py           # FastAPI app entry point, router registration, lifespan
│   ├── config/
│   │   └── settings.py   # All settings loaded from environment variables
│   ├── database/
│   │   └── session.py    # DB engine, session factory, table creation
│   ├── models/           # SQLModel table definitions (map to DB tables)
│   │   ├── user.py
│   │   ├── video.py
│   │   └── transcript.py
│   ├── schemas/          # Pydantic schemas for API request/response
│   │   ├── user.py
│   │   ├── video.py
│   │   └── transcript.py
│   ├── routers/          # FastAPI route handlers (thin — delegate to services)
│   │   ├── health.py
│   │   ├── auth.py
│   │   └── videos.py
│   ├── services/         # Business logic (sits between routers and DB)
│   │   ├── user_service.py
│   │   └── video_service.py
│   └── core/             # Shared utilities: security, dependencies, exceptions
├── tests/
│   ├── __init__.py
│   └── test_health.py
├── .env.example          # Template for environment variables
├── requirements.txt      # Python dependencies
└── README.md
```

### Why this structure?

| Folder | Purpose |
|---|---|
| `models/` | Defines database tables using SQLModel |
| `schemas/` | Defines what the API accepts and returns (separate from DB models) |
| `routers/` | HTTP layer only — validates input, calls a service, returns a response |
| `services/` | All business logic — routers stay clean, logic stays testable |
| `database/` | Engine setup and session management in one place |
| `config/` | Single source of truth for all configuration |
| `core/` | Shared tools used across features (auth helpers, custom errors, etc.) |

---

## Prerequisites

- **Python 3.14** (or 3.11+)
- PostgreSQL running locally (or via Docker)

---

## Local Setup

Run each command separately (one at a time).

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

Then activate it:

- **macOS / Linux:** `source venv/bin/activate`
- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Windows (CMD):** `venv\Scripts\activate.bat`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your `DATABASE_URL`:

```
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/studytube
```

### 4. Create the database

Using psql or any PostgreSQL client:

```sql
CREATE DATABASE studytube;
```

### 5. Run the development server

```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://localhost:8000`

---

## API Docs

FastAPI generates interactive docs automatically:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Key Endpoints (current)

| Method | Path | Description |
|---|---|---|
| GET | `/` | Root — confirms API is running |
| GET | `/health` | Health check for monitoring |
| GET | `/api/v1/auth/` | Auth placeholder |
| GET | `/api/v1/videos/` | Videos placeholder |

---

## Running Tests

```bash
pytest tests/
```

---

## Roadmap

- [ ] User registration and JWT authentication
- [ ] Video file upload and storage
- [ ] Transcript extraction (Whisper integration)
- [ ] RAG: embed transcript chunks and answer questions
- [ ] Summaries, quizzes, and flashcards generation
- [ ] Alembic database migrations
