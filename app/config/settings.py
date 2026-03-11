from pydantic_settings import BaseSettings


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

    # --- Security (placeholder for future JWT auth) ---
    SECRET_KEY: str = "changeme"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        # Tells pydantic-settings to look for a .env file
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single shared instance — import this wherever you need settings.
settings = Settings()
