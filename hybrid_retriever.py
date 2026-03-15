import yaml
from typing import List, Dict, Any
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from langchain_core.retrievers import BaseRetriever

class CustomHybridRetriever(BaseRetriever):
    """Custom Hybrid Retriever using Reciprocal Rank Fusion."""
    bm25_retriever: BM25Retriever
    vector_retriever: BaseRetriever
    weights: List[float] = Field(default_factory=lambda: [0.5, 0.5])
    c: int = 60
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        
        # Merge docs using RRF
        rrf_scores: Dict[str, Dict[str, Any]] = {}
        for weight, docs in zip(self.weights, [bm25_docs, vector_docs]):
            for rank, doc in enumerate(docs):
                # We use page_content + source as a unique key for deduplication
                doc_key = f"{doc.page_content}_{doc.metadata.get('source', '')}"
                if doc_key not in rrf_scores:
                    rrf_scores[doc_key] = {"score": 0.0, "doc": doc}
                rrf_scores[doc_key]["score"] += weight / (rank + 1 + self.c)
                
        # Sort and return top-k
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:self.bm25_retriever.k]]

def get_hybrid_retriever(vector_store: Chroma, documents: List[Document], k: int = 4):
    """
    Create a custom hybrid retriever combining BM25 and Vector search.
    """
    if not documents:
        return vector_store.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    return CustomHybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=[0.5, 0.5]
    )

def load_prompts(config_path: str = "config/prompts.yaml"):
    """Load prompts from YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    print("Hybrid retriever module ready.")
