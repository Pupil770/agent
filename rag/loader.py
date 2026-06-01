import os
from pathlib import Path

from langchain_core.documents import Document


def load_file(file_path: str, filename: str | None = None) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    if filename is None:
        filename = os.path.basename(file_path)

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(file_path).load()
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        docs = Docx2txtLoader(file_path).load()
    elif ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        docs = TextLoader(file_path, encoding="utf-8").load()
    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        docs = CSVLoader(file_path, encoding="utf-8").load()
    elif ext == ".xlsx":
        docs = _load_xlsx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    for doc in docs:
        doc.metadata["source"] = filename
    return docs


def _load_xlsx(file_path: str) -> list[Document]:
    import pandas as pd

    df = pd.read_excel(file_path)
    # 尝试识别 FAQ 格式：question/answer 列
    q_col = next((c for c in df.columns if "question" in c.lower() or "问题" in c), None)
    a_col = next((c for c in df.columns if "answer" in c.lower() or "回答" in c or "答案" in c), None)

    docs = []
    if q_col and a_col:
        for _, row in df.iterrows():
            content = f"Q: {row[q_col]}\nA: {row[a_col]}"
            docs.append(Document(page_content=content))
    else:
        for _, row in df.iterrows():
            content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
            docs.append(Document(page_content=content))
    return docs
