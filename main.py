from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import os
import re
import shutil

import boto3
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

DOCUMENTS_FILE = Path("documents.json")
ALLOWED_FILE_TYPES = {".pdf", ".txt"}

API_KEY = os.getenv("API_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key is not configured")

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def load_documents() -> list[dict]:
    if not DOCUMENTS_FILE.exists():
        return []

    return json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8"))


def save_documents(documents: list[dict]) -> None:
    DOCUMENTS_FILE.write_text(
        json.dumps(documents, indent=2),
        encoding="utf-8",
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
        text,
    )

    text = re.sub(
        r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[REDACTED_PHONE]",
        text,
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
    return {"message": "Clinical Document Intelligence API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/documents")
def get_documents(x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    return {"documents": load_documents()}


@app.get("/documents/{document_id}")
def get_document(document_id: str, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    documents = load_documents()

    for document in documents:
        if document["document_id"] == document_id:
            return document

    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)

    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3 bucket name is not configured")

    original_filename = file.filename

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must have a filename.",
        )

    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are allowed.",
        )

    document_id = str(uuid4())
    saved_filename = f"{document_id}{file_extension}"
    saved_path = UPLOAD_FOLDER / saved_filename

    with saved_path.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)

    s3_key = f"uploads/{saved_filename}"

    s3_client.upload_file(
        str(saved_path),
        S3_BUCKET_NAME,
        s3_key,
    )

    extracted_text = extract_text_from_file(saved_path)
    redacted_text = redact_pii(extracted_text)
    chunks = chunk_text(redacted_text)

    metadata = {
        "document_id": document_id,
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "file_type": file_extension,
        "s3_bucket": S3_BUCKET_NAME,
        "s3_key": s3_key,
        "character_count": len(extracted_text),
        "chunk_count": len(chunks),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    documents = load_documents()
    documents.append(metadata)
    save_documents(documents)

    return {
        "message": "Document uploaded to S3 and metadata saved successfully",
        **metadata,
        "redacted_preview": redacted_text[:500],
        "first_chunk": chunks[0] if chunks else "",
    }