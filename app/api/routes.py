from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents 
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


def _validate_ingest_access(x_admin_key: str) -> None:
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Document upload is not configured")
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")


def _normalize_upload_filename(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    safe_name = Path((filename or "").replace("\\", "/")).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="A valid filename is required")
    return safe_name


def _validate_upload_content(content: bytes) -> None:
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Maximum upload size is {settings.max_upload_mb} MB")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")


def _ensure_upload_destination_is_new(destination: Path) -> None:
    if destination.exists():
        raise HTTPException(status_code=409, detail="A file with this name is already uploaded")


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),

        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    _validate_ingest_access(x_admin_key)
    filename = _normalize_upload_filename(file.filename)
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    _validate_upload_content(content)
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / filename
    _ensure_upload_destination_is_new(dest)
    dest.write_bytes(content)
    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}
