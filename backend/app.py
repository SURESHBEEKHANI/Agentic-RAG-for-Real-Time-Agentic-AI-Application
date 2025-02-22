import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # Updated import
from langchain_core.messages import HumanMessage
from src.function import (
    ai_assistant,
    retrieve,  # Make sure this matches the variable name in function.py
    generate,
    rewrite,
    AgentState,
    grade_documents,
)
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

# Initialize FastAPI
app = FastAPI(
    title="Agentic RAG AI Assistant",
    description="An AI assistant powered by RAG and LangGraph",
    version="1.0.0"
)

# Request model
class Query(BaseModel):
    message: str
    
# Response model
class Response(BaseModel):
    response: str

# Initialize workflow
workflow = StateGraph(AgentState)
workflow.add_node("My_Ai_Assistant", ai_assistant)
workflow.add_node("Vector_Retriever", retrieve)
workflow.add_node("Output_Generator", generate)
workflow.add_node("Query_Rewriter", rewrite)

# Define edges
workflow.add_edge(START, "My_Ai_Assistant")
workflow.add_conditional_edges(
    "My_Ai_Assistant",
    tools_condition,
    {"tools": "Vector_Retriever", END: END},
)
workflow.add_conditional_edges(
    "Vector_Retriever",
    grade_documents,
    {"generator": "Output_Generator", "rewriter": "Query_Rewriter"},
)
workflow.add_edge("Output_Generator", END)
workflow.add_edge("Query_Rewriter", "My_Ai_Assistant")

# Compile workflow
app_workflow = workflow.compile()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to Agentic RAG AI Assistant API"
    }

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    try:
        # Create message
        message = HumanMessage(content=query.message)
        
        # Invoke workflow
        result = app_workflow.invoke({
            "messages": [message]
        })
        
        # Extract response
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            response_content = result["messages"][-1].content
            return Response(response=response_content)
        else:
            raise HTTPException(
                status_code=500,
                detail="Invalid response format from workflow"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# Swagger UI configuration
app.title = "Agentic RAG AI Assistant"
app.description = """
An AI assistant that uses Retrieval Augmented Generation (RAG) 
to provide informed responses based on specific document context.
"""
app.version = "1.0.0"
