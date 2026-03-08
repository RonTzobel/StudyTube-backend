from fastapi import APIRouter, HTTPException, UploadFile, File

from app.config.settings import settings
from app.services.video_service import save_upload_file

# Allowed video MIME types.
# This is a basic whitelist — we reject anything that doesn't look like a video.
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/webm",
    "video/ogg",
    "video/quicktime",  # .mov
    "video/x-msvideo",  # .avi
}

# 500 MB limit — prevents the server from being overwhelmed by huge files.
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.post("/upload")
def upload_video(file: UploadFile = File(...)):
    """
    Accept a video file from the client and save it to the uploads folder.

    How it works:
      1. Validate the file's MIME type (must be a video format we support).
      2. Validate the file size (must not exceed MAX_FILE_SIZE_BYTES).
      3. Delegate the actual saving to the service layer.
      4. Return the saved filename and a status message.

    Note: 'file: UploadFile = File(...)' tells FastAPI this endpoint
    expects a multipart/form-data request with a field named 'file'.
    The '...' means the field is required.

    Future improvements:
      - Store the file path in the database (linked to a Video record).
      - Move the file to cloud storage (S3, GCS) instead of local disk.
      - Trigger an async transcription job after saving.
    """
    # Step 1 — check the MIME type
    # upload_file.content_type is set by the client's browser or HTTP library.
    # It is not 100% trustworthy, but it is a good first line of defence.
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # Step 2 — check the file size
    # We read the first byte past the limit to detect oversized files
    # without loading the whole file into memory.
    file.file.seek(0, 2)          # seek to the end of the file
    file_size = file.file.tell()  # the position at the end = total size in bytes
    file.file.seek(0)             # seek back to the start before saving

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is 500 MB.",
        )

    # Step 3 — save the file via the service layer
    result = save_upload_file(
        upload_file=file,
        upload_dir=settings.UPLOAD_DIR,
    )

    # Step 4 — return a response the client can use
    return {
        "status": "uploaded",
        "filename": result["saved_filename"],
        "original_filename": file.filename,
        "message": "File saved successfully. Transcription will be triggered in a future step.",
    }


@router.get("/")
def list_videos_placeholder():
    """Placeholder — will list the current user's videos once auth is in place."""
    return {"message": "Videos router is ready. Implementation coming soon."}
