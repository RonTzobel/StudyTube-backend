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

    # --- Database ---
    # Full PostgreSQL connection string.
    # Example: postgresql://user:password@localhost:5432/studytube
    DATABASE_URL: str

    # --- File uploads ---
    # Directory where uploaded video files are temporarily stored.
    # In production this would point to a cloud bucket or a mounted volume.
    UPLOAD_DIR: str = "uploads"

    # --- OpenAI ---
    # Required for the /ask endpoint (RAG answer generation).
    # Get your key at https://platform.openai.com/api-keys
    # Keep this out of version control — set it only in your .env file.
    OPENAI_API_KEY: str = ""

    # --- RAG quality gate (two-level policy) ---
    #
    # We use two thresholds instead of one rigid cutoff because
    # all-MiniLM-L6-v2 is English-optimised. When it encodes Hebrew text it
    # still produces meaningful vectors, but scores are systematically
    # 30–50 % lower than for equivalent English content.
    # A relevant Hebrew question typically scores 0.12 – 0.35 (vs 0.40 – 0.75
    # for English), so a single threshold of 0.30 rejects too many valid questions.
    #
    # Policy:
    #   best_score < RAG_LOW_THRESHOLD   → immediate fallback, no OpenAI call
    #                                      (score is so low the content is
    #                                       almost certainly off-topic)
    #   RAG_LOW_THRESHOLD ≤ score
    #         < RAG_GOOD_THRESHOLD       → call OpenAI with the same strict
    #                                      grounded prompt; response is marked
    #                                      confidence_level="low"
    #   best_score ≥ RAG_GOOD_THRESHOLD  → normal grounded flow;
    #                                      confidence_level="high"
    #
    # Range for both values: 0.0 – 1.0
    RAG_LOW_THRESHOLD: float = 0.10
    RAG_GOOD_THRESHOLD: float = 0.22

    # --- Whisper transcription ---
    # Model size: "medium" is recommended for Hebrew (small/base degrade noticeably).
    # device="cpu" and compute_type="int8" are safe defaults for Windows / no-GPU setups.
    WHISPER_MODEL: str = "medium"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # --- Security (placeholder for future JWT auth) ---
    SECRET_KEY: str = "changeme"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- Google OAuth ---
    # Create credentials at https://console.cloud.google.com/apis/credentials
    # Set the redirect URI to: http://127.0.0.1:8000/api/v1/auth/google/callback
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://127.0.0.1:8000/api/v1/auth/google/callback"

    # --- Frontend ---
    # After Google OAuth, the backend redirects here with ?token=<jwt>.
    # The frontend /auth/callback page reads the token and stores it.
    FRONTEND_URL: str = "http://localhost:5173"

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"


# Single shared instance — import this wherever you need settings.
settings = Settings()
