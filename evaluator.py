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
faithfulness = Faithfulness()
answer_relevance = AnswerRelevancy()
context_precision = ContextPrecision()
context_recall = ContextRecall()
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
        
        # Note: RAGAS might need specific embeddings too.
        # We'll use HuggingFace embeddings which are free/local.
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
            metrics=[
                faithfulness,
                answer_relevance,
                context_precision,
            ],
            # llm=self.judge_llm, # RAGAS expects a specific LLM wrapper, might need adaptation
            # embeddings=self.embeddings
        )
        
        df = score.to_pandas()
        print("\nEvaluation Results:")
        print(df)
        
        # Log scores to Langfuse (using the overall average for now or per-trace)
        for metric_name, value in score.items():
            print(f"Metric {metric_name}: {value:.4f}")
            log_metric(f"ragas_{metric_name}", value)
            
        return score

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    # Example bench
    questions = ["What is the main argument of Progress and Poverty?"]
    evaluator.run_evaluation(questions)
