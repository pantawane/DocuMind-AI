from pathlib import Path
import re
from typing import Iterable
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from docx import Document as DocxDocument


SUPPORTED = {".pdf", ".txt", ".md", ".docx"}



def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": str(path)})]
    raise ValueError(f"Unsupported file type: {suffix}")



def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120, add_start_index=True)
    chunks: list[Document] = []
    for doc in docs:
        doc_chunks = splitter.split_documents([doc])
        headings = list(re.finditer(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", doc.page_content, re.MULTILINE))

        # Keep the nearest preceding Markdown heading with each chunk so answers
        # can point people to a useful section, even when the heading is in the
        # previous chunk.
        for chunk in doc_chunks:
            start_index = chunk.metadata.get("start_index")
            if start_index is not None:
                preceding = [heading for heading in headings if heading.start() <= start_index]
                if preceding:
                    chunk.metadata["section"] = preceding[-1].group(1).strip()
        chunks.extend(doc_chunks)

    return chunks
