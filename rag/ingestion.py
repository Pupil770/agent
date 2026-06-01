import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.store import get_vectorstore


def ingest_documents(docs: list, doc_id: str | None = None) -> str:
    doc_id = doc_id or str(uuid.uuid4())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id

    get_vectorstore().add_documents(chunks)
    return doc_id


def delete_documents(doc_id: str) -> int:
    vs = get_vectorstore()
    collection = vs._collection
    result = collection.get(where={"doc_id": doc_id})
    ids = result["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_documents() -> list[dict]:
    vs = get_vectorstore()
    collection = vs._collection
    all_data = collection.get(include=["metadatas"])
    seen = {}
    for meta in all_data["metadatas"]:
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "source": meta.get("source", "unknown")}
    return list(seen.values())