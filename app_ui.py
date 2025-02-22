import streamlit as st
import requests
import json
from typing import List, Dict

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def display_chat_history():
    """Display chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_message(role: str, content: str):
    """Add a message to the chat history"""
    st.session_state.messages.append({"role": role, "content": content})

def query_assistant(prompt: str) -> str:
    """Send query to FastAPI backend"""
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"message": prompt},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with backend: {str(e)}")
        return None

# UI Elements
st.title("🤖 Agentic RAG Assistant")

st.markdown("""
This AI assistant uses Retrieval Augmented Generation (RAG) to provide informed responses 
based on specific document context. It can answer questions about:
- LLM Agents
- Prompt Engineering
- AI Systems
""")

# Display chat history
display_chat_history()

# Chat input
if prompt := st.chat_input("Ask me anything about AI agents or prompt engineering!"):
    # Display user message
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if response := query_assistant(prompt):
                add_message("assistant", response)
                st.markdown(response)
            else:
                st.error("Failed to get response from assistant")

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This assistant is powered by:
    - LangChain
    - LangGraph
    - Groq LLM
    - ChromaDB
    - FastAPI
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
