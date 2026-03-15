import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from dotenv import load_dotenv
import time

# Local imports
from rag_system import RAGSystem
from evaluator import RAGEvaluator
from observability import log_metric, observe, get_client

# Page config
st.set_page_config(
    page_title="Production RAG Dashboard",
    page_icon="🤖",
    layout="wide",
)

# Load env vars
load_dotenv()

# CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .stMetric {
        background-color: #1e2227;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4451;
    }
    .stChatMessage {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_history" not in st.session_state:
    st.session_state.eval_history = []

@st.cache_resource
def get_systems():
    """Initialize RAG and Evaluator once."""
    rag = RAGSystem()
    evaluator = RAGEvaluator()
    return rag, evaluator

rag, evaluator = get_systems()

# Sidebar
with st.sidebar:
    st.title("🛡️ RAG Observability")
    st.markdown("---")
    st.info("System is monitoring performance, costs, and quality metrics in real-time.")
    
    # Langfuse deep link (placeholder)
    st.markdown("### 📊 Tracing")
    if os.getenv("LANGFUSE_HOST"):
        st.link_button("View Langfuse Traces", os.getenv("LANGFUSE_HOST"))
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_retrieved = st.checkbox("Show Retrieved Context", value=True)
    run_eval = st.checkbox("Run RAGAS Evaluation", value=True)

# Main Header
st.title("🚀 Production RAG Dashboard")
st.caption("Monitoring & Observability for high-stakes AI applications")

# Analytics Section (at the top)
if st.session_state.eval_history:
    st.subheader("📈 Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    df_history = pd.DataFrame(st.session_state.eval_history)
    
    with col1:
        avg_faith = df_history["faithfulness"].mean()
        st.metric("Faithfulness", f"{avg_faith:.2f}", delta=None)
    with col2:
        avg_relevance = df_history["answer_relevancy"].mean()
        st.metric("Answer Relevancy", f"{avg_relevance:.2f}", delta=None)
    with col3:
        avg_precision = df_history["context_precision"].mean()
        st.metric("Context Precision", f"{avg_precision:.2f}", delta=None)

    # Plotly Chart
    fig = go.Figure()
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    for m in metrics:
        fig.add_trace(go.Scatter(
            y=df_history[m], 
            mode='lines+markers', 
            name=m.capitalize().replace("_", " "),
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title="Metric Evolution",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "eval_scores" in message:
            with st.expander("🔍 Evaluation Details"):
                st.json(message["eval_scores"])

# Handle Query
if prompt := st.chat_input("Ask about Progress and Poverty..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and Evaluating..."):
            start_time = time.time()
            
            # 1. Query RAG
            # We use the existing observabilty-decorated query
            answer, docs = rag.query(prompt)
            
            end_time = time.time()
            latency = end_time - start_time
            
            st.markdown(answer)
            
            # 2. Results and Metadata
            eval_scores = {}
            if run_eval:
                # 3. Run RAGAS
                score = evaluator.run_evaluation([prompt])
                # Access scores from EvaluationResult.to_pandas() as we found earlier
                df_score = score.to_pandas()
                eval_scores = df_score.mean(numeric_only=True).to_dict()
                
                # Update session history
                st.session_state.eval_history.append({
                    "timestamp": datetime.now(),
                    "latency": latency,
                    **eval_scores
                })
            
            # Show context if requested
            if show_retrieved:
                with st.expander("📄 Retrieved Context"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content[:500] + "...")
            
            # Save for UI persistence
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "eval_scores": eval_scores
            })
            
            # Rerender to show graphs update
            st.rerun()
