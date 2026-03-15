from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Initialize the cross-encoder model."""
    return CrossEncoder(model_name)

def rerank_documents(query: str, documents: List[Document], model: CrossEncoder, top_n: int = 3) -> List[Document]:
    """
    Rerank documents based on the query using a cross-encoder.
    """
    if not documents:
        return []
        
    doc_texts = [doc.page_content for doc in documents]
    # Cross-encoder expects pairs of (query, doc_text)
    pairs = [(query, text) for text in doc_texts]
    
    scores = model.predict(pairs)
    
    # Sort docs by score
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    # Return top_n documents
    return [doc for _, doc in scored_docs[:top_n]]

if __name__ == "__main__":
    print("Reranker module ready.")
