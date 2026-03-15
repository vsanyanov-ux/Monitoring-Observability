import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langfuse import observe, get_client

# Local imports
from loader import load_pdf
from splitter import split_documents
from vector_store import get_vector_store, add_documents_to_store
from hybrid_retriever import get_hybrid_retriever, load_prompts
from reranker import get_reranker, rerank_documents
from observability import trace_step

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.yc_api_key = os.getenv("YC_API_KEY")
        self.yc_folder_id = os.getenv("YC_FOLDER_ID")
        
        self.llm = ChatYandexGPT(
            api_key=self.yc_api_key,
            folder_id=self.yc_folder_id,
            model_uri=f"gpt://{self.yc_folder_id}/yandexgpt/latest",
            temperature=0.1
        )
        
        prompts = load_prompts()
        self.prompt_temp = PromptTemplate.from_template(
            f"{prompts.get('system_prompt', '')}\n\n{prompts.get('qa_template', '')}"
        )
        self.reranker_model = get_reranker()

    @observe()
    def ingest(self, path: str):
        """Index documents with tracing."""
        docs = load_pdf(path)
        chunks = split_documents(docs)
        store = get_vector_store()
        add_documents_to_store(store, chunks)
        return len(chunks)

    @observe()
    @trace_step("Retrieval & Reranking")
    def retrieve(self, question: str, top_k: int = 10, top_n: int = 3):
        store = get_vector_store()
        retriever = store.as_retriever(search_kwargs={"k": top_k})
        
        # Initial retrieval
        initial_docs = retriever.invoke(question)
        
        # Reranking
        final_docs = rerank_documents(question, initial_docs, self.reranker_model, top_n=top_n)
        
        contexts = [doc.page_content for doc in final_docs]
        return contexts, final_docs

    @observe()
    @trace_step("Generation")
    def generate(self, question: str, contexts: list):
        chain = self.prompt_temp | self.llm | StrOutputParser()
        context_text = "\n\n".join(contexts)
        
        # We can pass more metadata to Langfuse via context if needed
        # langfuse_context.update_current_observation(input={"question": question})
        
        answer = chain.invoke({"context": context_text, "question": question})
        return answer

    @observe(name="RAG Query")
    def query(self, question: str):
        """Execute the full RAG pipeline."""
        contexts, docs = self.retrieve(question)
        answer = self.generate(question, contexts)
        
        get_client().update_current_span(output=answer)
        return answer, docs

if __name__ == "__main__":
    rag = RAGSystem()
    print("RAG System with Observability Ready.")
