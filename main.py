from pathlib import Path
from uuid import uuid4
import os
import re
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pypdf import PdfReader


load_dotenv()

app = FastAPI(
    title="Clinical Document Intelligence API",
    version="1.0.0",
)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_FILE_TYPES = {".pdf", ".txt"}

API_KEY = os.getenv("API_KEY")


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key is not configured on the server"
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


def extract_text_from_file(file_path: Path) -> str:
    file_extension = file_path.suffix.lower()

    if file_extension == ".txt":
        return file_path.read_text(encoding="utf-8")

    if file_extension == ".pdf":
        reader = PdfReader(str(file_path))
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        return "\n".join(pages_text)

    raise ValueError("Unsupported file type")


def redact_pii(text: str) -> str:
    text = re.sub(
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        "[REDACTED_EMAIL]",
        text
    )

    text = re.sub(
        r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[REDACTED_PHONE]",
        text
    )

    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


@app.get("/")
def home():
    return {
        "message": "Clinical Document Intelligence API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None)
):
    verify_api_key(x_api_key)

    original_filename = file.filename

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must have a filename."
        )

    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are allowed."
        )

    saved_filename = f"{uuid4()}{file_extension}"
    saved_path = UPLOAD_FOLDER / saved_filename

    with saved_path.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)

    extracted_text = extract_text_from_file(saved_path)
    redacted_text = redact_pii(extracted_text)
    chunks = chunk_text(redacted_text)

    return {
        "message": "Document uploaded, text extracted, PII redacted, and chunked successfully",
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "file_type": file_extension,
        "character_count": len(extracted_text),
        "chunk_count": len(chunks),
        "redacted_preview": redacted_text[:500],
        "first_chunk": chunks[0] if chunks else ""
    }