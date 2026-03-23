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
    ENV: str = "development"

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
    # Current default: paraphrase-multilingual-MiniLM-L12-v2
    #   - 384-dimensional vectors (same storage format as the previous model)
    #   - Trained on 50+ languages; strong Hebrew support
    #   - ~470 MB on disk; runs on CPU
    #   - Typical cosine scores for relevant Hebrew pairs: 0.40 – 0.80
    #
    # Previous model (English-biased): all-MiniLM-L6-v2
    #   - 384-dimensional vectors
    #   - Hebrew pairs scored 0.12 – 0.35 (30–50 % lower than English)
    #   - ~80 MB on disk
    #
    # To switch models: change this value AND re-process all existing videos
    # (POST /api/v1/videos/{id}/transcribe for each) so chunks are re-embedded
    # with the new model. Do NOT mix chunk embeddings from different models.
    EMBEDDING_MODEL_NAME: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --- RAG quality gate (two-level policy) ---
    #
    # Calibrated for paraphrase-multilingual-MiniLM-L12-v2.
    # This model produces much higher cosine scores than the previous
    # English-only model — relevant Hebrew pairs now score 0.40 – 0.80,
    # so the old thresholds of 0.10 / 0.22 were far too low.
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
    # If you switch back to all-MiniLM-L6-v2, lower these to 0.10 / 0.22.
    RAG_LOW_THRESHOLD: float = 0.20
    RAG_GOOD_THRESHOLD: float = 0.40

    # --- Whisper transcription ---
    #
    # Speed / quality guide for Hebrew lecture content on CPU:
    #
    #   WHISPER_MODEL — controls the size of the transcription model.
    #     "base"    fastest  (~3–6 min per 30-min video on CPU)
    #               Hebrew accuracy noticeably lower, may miss words/phrases.
    #     "small"   balanced (~8–15 min per 30-min video on CPU)  ← default
    #               Good Hebrew accuracy for lecture content.
    #     "medium"  highest quality (~30–60 min per 30-min video on CPU)
    #               Impractical for long videos on CPU. Use with GPU only.
    #
    #   WHISPER_BEAM_SIZE — beam search width during decoding.
    #     1   greedy decoding: 2–3× faster, minimal quality loss for structured
    #         lecture speech. Recommended for CPU.          ← default
    #     5   Whisper's original default: better recovery of ambiguous words,
    #         but significantly slower. Use for quality-critical transcriptions.
    #
    #   WHISPER_VAD_FILTER — Voice Activity Detection preprocessing.
    #     True   skip silence and non-speech frames before transcription.
    #            Saves 10–30 % of processing time for typical lecture videos
    #            with pauses, intros, or Q&A gaps.          ← default
    #     False  process all audio including silence.
    #
    #   WHISPER_CPU_THREADS — CTranslate2 intra-op thread count.
    #     0     let CTranslate2 pick automatically (usually uses all cores). ← default
    #     N     force N threads; try 4–8 on modern multi-core CPUs.
    #
    # To restore the old quality-first settings, add to your .env:
    #   WHISPER_MODEL=medium
    #   WHISPER_BEAM_SIZE=5
    #
    WHISPER_MODEL: str = "small"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 1
    WHISPER_VAD_FILTER: bool = True
    WHISPER_CPU_THREADS: int = 0

    # --- CORS ---
    # List of allowed origins for the frontend.
    # pydantic-settings parses a JSON array string from the .env file:
    #   CORS_ALLOWED_ORIGINS=["https://app.studytube.com"]
    # Default covers both Vite dev-server variants for local development.
    CORS_ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # --- Redis ---
    # Local:  redis://localhost:6379/0
    # Docker: redis://redis:6379/0  (matches the service name in docker-compose)
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Worker job timeouts ---
    # Maximum wall-clock seconds a video pipeline job may run before RQ kills it.
    # Default: 7200 (2 hours) — generous ceiling for a 90-min lecture on CPU.
    # Increase if you process very long videos on slow hardware.
    VIDEO_PIPELINE_JOB_TIMEOUT: int = 7200

    # --- Security ---
    SECRET_KEY: str = "changeme"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

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
