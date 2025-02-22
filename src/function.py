import os
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field  # Updated import

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Import necessary modules from LangChain, LangGraph, and related libraries
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Initialize embeddings and the language model (LLM)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="Gemma2-9b-It")

# ---------------------------
# Load and Process Web Documents
# ---------------------------

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

# Load documents from each URL
docs = [WebBaseLoader(url).load() for url in URLS]
# Flatten the list of document lists into a single list
docs_list = [doc for sublist in docs for doc in sublist]

# Split documents into chunks for efficient processing
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=25
)
doc_splits = text_splitter.split_documents(docs_list)

# Create a vector store using the document splits and HuggingFace embeddings
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chrome",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# Create a retriever tool specific to the blog posts domain
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    (
        "Search and return information about Lilian Weng blog posts on LLM agents, "
        "prompt engineering, and adversarial attacks on LLMs. You are a specialized assistant. "
        "Use the 'retriever_tool' **only** when the query explicitly relates to LangChain blog data. "
        "For all other queries, respond directly without using any tool. For simple queries like "
        "'hi', 'hello', or 'how are you', provide a normal response."
    ),
)
tools = [retriever_tool]
retrieve = ToolNode([retriever_tool])  # This is what we'll export

# ---------------------------
# Define Agent State and Functions
# ---------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def ai_assistant(state: AgentState) -> dict:
    """
    Processes the conversation state and generates a response using the LLM.
    If there is more than one message, it uses a prompt template to generate an answer.
    Otherwise, it binds available tools to the LLM before invocation.
    """
    print("---CALL AGENT---")
    messages = state["messages"]

    if len(messages) > 1:
        # Use the latest message's content as the query
        question = messages[-1].content
        prompt = PromptTemplate(
            template=(
                "You are a helpful assistant. Analyze the following question and provide a detailed answer.\n"
                "Question: {question}"
            ),
            input_variables=["question"],
        )
        chain = prompt | llm
        response = chain.invoke({"question": question})
        return {"messages": [response]}
    else:
        # Bind the retriever tool to the LLM and process the initial message
        llm_with_tool = llm.bind_tools(tools)
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}


class GradeOutput(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")


def grade_documents(state: AgentState) -> Literal["generator", "rewriter"]:
    """
    Grades the relevance of a document with respect to a user's question.
    Returns 'generator' if the document is relevant and 'rewriter' otherwise.
    """
    llm_with_structure = llm.with_structured_output(GradeOutput)
    prompt = PromptTemplate(
        template=(
            "You are a grader deciding if a document is relevant to a user’s question.\n"
            "Document: {context}\n"
            "User’s question: {question}\n"
            "If the document contains information related to the question, answer 'yes', otherwise 'no'."
        ),
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_structure

    messages = state["messages"]
    question = messages[0].content
    document_content = messages[-1].content

    scored_result = chain.invoke({"question": question, "context": document_content})
    score = scored_result.binary_score.lower()

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generator"  # Node name for generating output
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewriter"  # Node name for rewriting query


def generate(state: AgentState) -> dict:
    """
    Generates a final answer by using a retrieval-augmented generation (RAG) prompt.
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    document_content = messages[-1].content

    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_chain = rag_prompt | llm
    response = rag_chain.invoke({"context": document_content, "question": question})
    print(f"Response: {response}")
    return {"messages": [response]}


def rewrite(state: AgentState) -> dict:
    """
    Rewrites the user's query by analyzing its semantic intent and providing an improved version.
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    prompt_message = HumanMessage(
        content=(
            f"Review the input and infer the underlying semantic intent.\n"
            f"Initial question: {question}\n"
            "Formulate an improved version of the question:"
        )
    )
    response = llm.invoke([prompt_message])
    return {"messages": [response]}
