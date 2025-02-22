# Agentic RAG AI Assistant

A real-time AI assistant that combines Large Language Models with Retrieval Augmented Generation (RAG) to provide context-aware, intelligent responses.

## Features

- 🤖 LLM-powered conversational agent
- 📚 Dynamic knowledge retrieval using RAG
- 🔄 Intelligent query rewriting
- 💡 Real-time response generation
- 🌐 Clean web interface using Streamlit
- 🚀 Fast API backend using FastAPI

## Architecture

The system consists of several key components:

1. **LangChain & LangGraph** for orchestration
2. **Groq LLM** for fast inference
3. **ChromaDB** for vector storage
4. **FastAPI** backend API
5. **Streamlit** frontend interface

## Setup

1. Clone the repository:
```bash
git clone https://github.com/SURESHBEEKHANI/Agentic-RAG-for-Real-Time-Agentic-AI-Application.git
cd Agentic-RAG-for-Real-Time-Agentic-AI-Application
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
GROQ_API_KEY=your_groq_api_key
```

4. Run the application:

Backend:
```bash
cd backend
python main.py
```

Frontend:
```bash
streamlit run app_ui.py
```

## Project Structure

```
├── app.py              # FastAPI application
├── app_ui.py          # Streamlit frontend
├── backend/
│   └── main.py        # Backend entry point
├── src/
│   └── function.py    # Core functionality
└── resources/
    └── agentic_rag.ipynb  # Jupyter notebook with development
```

## How It Works

1. **Query Processing**: User inputs are processed through a sophisticated workflow
2. **Context Retrieval**: Relevant context is retrieved using RAG
3. **Response Generation**: Context-aware responses are generated
4. **Query Rewriting**: Queries can be rewritten for better results

## API Endpoints

- `GET /`: Health check endpoint
- `POST /query`: Main query endpoint
  - Request body: `{"message": "your question here"}`
  - Response: `{"response": "AI assistant response"}`


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with LangChain and LangGraph
- Powered by Groq LLM
- Vector storage by ChromaDB
- Web framework by FastAPI
- UI powered by Streamlit

