import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_markdown(file_path: str) -> List[Document]:
    """Load a Markdown file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()

def load_web_url(url: str) -> List[Document]:
    """Load a document from a web URL."""
    loader = WebBaseLoader(url)
    return loader.load()

if __name__ == "__main__":
    # Quick test if needed
    print("Loader utility ready.")
