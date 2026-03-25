from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve the .env file relative to this file's location so the path is
# always correct regardless of the working directory at launch time.
# settings.py → app/config/   parent → app/   parent → backend/  ← .env lives here
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    pydantic-settings automatically reads values from a .env file
    or from real environment variables. If a value is missing and
    has no default, the app will raise an error at startup — which
    is exactly what we want (fail fast, never run with missing config).
    """

    # --- App ---
    APP_NAME: str = "StudyTube"
    DEBUG: bool = False

    # Runtime environment — used for conditional behaviour and logging.
    # Accepted values: "development" | "production" | "test"
    ENV: str = "production"

    # --- Database ---
    # Full PostgreSQL connection string.
    # Example: postgresql://user:password@localhost:5432/studytube
    DATABASE_URL: str

    # --- File uploads ---
    # Directory where uploaded video files are temporarily stored.
    # In production this would point to a cloud bucket or a mounted volume.
    UPLOAD_DIR: str = "uploads"

    # --- AWS S3 ---
    # Used for video file storage. Videos are never stored in PostgreSQL —
    # only the S3 key (object path) is saved in the database.
    #
    # Required IAM permissions for the configured user:
    #   s3:PutObject, s3:DeleteObject, s3:GetObject on arn:aws:s3:::{bucket}/*
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = ""
    AWS_S3_BUCKET: str = ""

    # --- OpenAI ---
    # Required for the /ask endpoint (RAG answer generation).
    # Get your key at https://platform.openai.com/api-keys
    # Keep this out of version control — set it only in your .env file.
    OPENAI_API_KEY: str = ""

    # --- Embedding model ---
    #
    # The sentence-transformers model used to embed transcript chunks and
    # user queries. Both MUST use the same model — mixing models produces
    # semantically incompatible vectors and silently breaks retrieval.
    #
    # BAAI/bge-small-en-v1.5: strong English retrieval, 384-dim, fast on CPU.
    # Good balance of quality vs. memory for EC2 CPU deployments.
    # Swap to paraphrase-multilingual-MiniLM-L12-v2 for Hebrew/multilingual.
    #
    # If you change this model, re-index all existing videos — old and new
    # vectors are geometrically incompatible and will silently break search.
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

    # --- RAG quality gate (two-level policy) ---
    # Calibrated for BAAI/bge-small-en-v1.5 cosine similarity score range.
    # Below LOW  → immediate fallback, no OpenAI call (not enough context)
    # Above GOOD → high-confidence answer
    # Between   → borderline (OpenAI called with wider chunk window)
    RAG_LOW_THRESHOLD: float = 0.30
    RAG_GOOD_THRESHOLD: float = 0.60

    # --- External transcription (OpenAI Whisper API) ---
    # Transcription is performed by the OpenAI Whisper API — no local model
    # is loaded in the worker. OPENAI_API_KEY (above) is reused.
    # Audio is split into 5-minute chunks before upload to stay within the
    # 25 MB per-request API limit.
    OPENAI_TRANSCRIPTION_MODEL: str = "whisper-1"
    # Per-chunk HTTP timeout in seconds. 5-minute audio chunks typically
    # complete in 10–30s; 300s gives ample headroom for slow API responses.
    TRANSCRIPTION_TIMEOUT: int = 300

    # --- CORS ---
    # Default now includes production domains + local dev.
    CORS_ALLOWED_ORIGINS: list[str] = [
        "https://studytubeapp.com",
        "https://www.studytubeapp.com",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # --- Redis ---
    # Local:  redis://localhost:6379/0
    # Docker: redis://redis:6379/0  (matches the service name in docker-compose)
    REDIS_URL: str = "redis://redis:6379/0"

    # --- Worker job timeouts ---
    VIDEO_PIPELINE_JOB_TIMEOUT: int = 7200

    # --- Security ---
    SECRET_KEY: str = "changeme"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    # --- Google OAuth ---
    # Production redirect URI must match Google Cloud Console exactly.
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = (
        "https://studytubeapp.com/api/v1/auth/google/callback"
    )

    # --- Frontend ---
    # After Google OAuth, the backend redirects here with ?token=<jwt>.
    # The frontend /auth/callback page reads the token and stores it.
    FRONTEND_URL: str = "https://studytubeapp.com"

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"


# Single shared instance — import this wherever you need settings.
settings = Settings()