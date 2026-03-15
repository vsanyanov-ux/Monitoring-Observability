import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse import Langfuse, observe

load_dotenv()

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

def trace_step(step_name: str):
    """Decorator to trace individual steps of the RAG pipeline."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                
                # Log to Langfuse if context is active
                # langfuse_context.update_current_span(name=step_name)
                
                print(f"DEBUG: Step '{step_name}' took {latency:.4f}s")
                return result
            except Exception as e:
                print(f"ERROR in {step_name}: {e}")
                raise e
        return wrapper
    return decorator

@observe()
def track_event(name: str, metadata: Dict[str, Any] = None):
    """Simple wrapper to track custom events or spans."""
    print(f"Event: {name}, Metadata: {metadata}")

def log_metric(name: str, value: float, trace_id: str = None):
    """Log a score/metric to Langfuse."""
    langfuse.score(
        name=name,
        value=value,
        trace_id=trace_id
    )

if __name__ == "__main__":
    print("Observability module initialized.")
