from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(
    documents: List[Document], 
    chunk_size: int = 1200, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks.
    Targeting 500-800 tokens. RecursiveCharacterTextSplitter uses characters by default,
    but we'll optimize for around that size in characters as a proxy for tokens 
    unless more precise token counting is required immediately.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    print("Splitter utility ready.")
