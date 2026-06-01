import os

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import VectorStore

_embedding = None
_vectorstore = None


def get_embedding() -> DashScopeEmbeddings:
    global _embedding
    if _embedding is None:
        _embedding = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        )
    return _embedding


def get_vectorstore() -> VectorStore:
    global _vectorstore
    if _vectorstore is None:
        backend = os.environ.get("VECTORSTORE_BACKEND", "chroma")
        if backend == "chroma":
            _vectorstore = _create_chroma()
        else:
            raise ValueError(f"Unsupported vectorstore backend: {backend}")
    return _vectorstore


def _create_chroma() -> VectorStore:
    from langchain_chroma import Chroma

    persist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_data")
    return Chroma(
        collection_name="knowledge_base",
        embedding_function=get_embedding(),
        persist_directory=persist_dir,
    )