import os

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from rag.ingestion import delete_documents, ingest_documents, list_documents
from rag.loader import load_file
from rag.store import get_vectorstore

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_name: str = Form(None),
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = doc_name or file.filename or "unnamed"
    dest = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    try:
        docs = load_file(dest, filename=safe_name)
        doc_id = ingest_documents(docs)
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise HTTPException(status_code=422, detail=f"Ingestion failed: {e}")

    return {"doc_id": doc_id, "filename": safe_name, "chunks": len(docs)}


@router.get("/documents")
async def get_documents():
    return {"documents": list_documents()}


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    count = delete_documents(doc_id)
    if count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted_chunks": count}


@router.post("/query")
async def query_knowledge(query: str = Query(...), top_k: int = Query(4)):
    vs = get_vectorstore()
    results = vs.similarity_search(query, k=top_k)
    return {
        "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    }