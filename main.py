import sys
import os
from dotenv import load_dotenv

# Local imports
from rag_system import RAGSystem
from evaluator import RAGEvaluator

load_dotenv()

def main():
    print("=== RAG Monitoring & Observability (Project 3) ===")
    
    rag = RAGSystem()
    
    # Check if data exists
    pdf_path = "data/Progress_and_Poverty.pdf"
    if os.path.exists(pdf_path):
        print(f"Ingesting {pdf_path}...")
        rag.ingest(pdf_path)
    else:
        print(f"Warning: {pdf_path} not found. Skipping ingestion.")

    print("\nRunning a sample query...")
    question = "How does Henry George explain the persistence of poverty?"
    answer, _ = rag.query(question)
    
    print("\nAI Answer:")
    print(answer)
    
    print("\nStarting automated evaluation...")
    evaluator = RAGEvaluator()
    eval_questions = [
        "What is the relationship between wages and interest according to George?",
        "Why does progress lead to poverty in George's view?"
    ]
    evaluator.run_evaluation(eval_questions)
    
    print("\nDone! Check Langfuse for traces and evaluation scores.")

if __name__ == "__main__":
    main()
