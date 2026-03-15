import os
import pandas as pd
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from datasets import Dataset

# Local imports
from rag_system import RAGSystem
from observability import log_metric

load_dotenv()

class RAGEvaluator:
    def __init__(self):
        # RAGAS often works best with OpenAI models for judging, 
        # but we can try to use YandexGPT if we configure it as a LangChain LLM for RAGAS.
        # For simplicity in this demo, we'll assume an OpenAI-compatible judge is available
        # or we'll wrap YandexGPT.
        from langchain_community.chat_models import ChatYandexGPT
        
        self.rag_system = RAGSystem()
        
        # Configure the judge LLM
        yc_api_key = os.getenv("YC_API_KEY")
        yc_folder_id = os.getenv("YC_FOLDER_ID")
        
        self.judge_llm = ChatYandexGPT(
            api_key=yc_api_key,
            folder_id=yc_folder_id,
            model_uri=f"gpt://{yc_folder_id}/yandexgpt/latest",
            temperature=0
        )
        
        # Wrap for RAGAS
        self.ragas_llm = LangchainLLMWrapper(self.judge_llm)
        
        # Note: RAGAS might need specific embeddings too.
        # We'll use HuggingFace embeddings which are free/local.
        from langchain_huggingface import HuggingFaceEmbeddings
        self.langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.langchain_embeddings)
        
        # Initialize metrics with the judge LLM
        self.metrics = [
            Faithfulness(llm=self.ragas_llm),
            AnswerRelevancy(llm=self.ragas_llm),
            ContextPrecision(llm=self.ragas_llm),
        ]

    def run_evaluation(self, test_questions: list, ground_truths: list = None):
        """Run RAGAS evaluation on a set of questions."""
        results = []
        
        print(f"Running evaluation on {len(test_questions)} questions...")
        
        for i, question in enumerate(test_questions):
            print(f"[{i+1}/{len(test_questions)}] Processing: {question}")
            answer, docs = self.rag_system.query(question)
            contexts = [doc.page_content for doc in docs]
            
            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truths[i] if ground_truths else ""
            })
            
        dataset = Dataset.from_list(results)
        
        # Run evaluation
        # We specify the metrics we want
        score = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )
        
        df = score.to_pandas()
        print("\nEvaluation Results:")
        print(df)
        
        # Log scores to Langfuse
        avg_scores = df.mean(numeric_only=True).to_dict()
        for metric_name, value in avg_scores.items():
            print(f"Metric {metric_name}: {value:.4f}")
            log_metric(f"ragas_{metric_name}", value)
            
        return score

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    # Example bench
    questions = ["What is the main argument of Progress and Poverty?"]
    evaluator.run_evaluation(questions)
