# RAG-based AI Chatbot  
**LangGraph · Pinecone · Groq · FastAPI**

**Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from :** [AGENTIC_AI_EBOOK](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf)
  
## Key Features

- PDF ingestion and semantic chunking  
- Vector-based semantic search (SentenceTransformers)  
- Transparent retrieval (context chunks + similarity scores)  
- Hallucination-safe answers (context-only)  
- REST API with Swagger UI (FastAPI)  
- CPU-only, no local LLMs required  

## Architecture Overview

This project implements a **Retrieval-Augmented Generation (RAG)** architecture.

User queries are first converted into vector embeddings using **SentenceTransformers** and matched against document embeddings stored in **Pinecone**. The most relevant chunks are retrieved using semantic similarity search.

The retrieval and generation flow is orchestrated using **LangGraph**, which cleanly separates the system into a **Retrieve** step and a **Generate** step. Retrieved context is then passed to a hosted **Groq (LLaMA 3.1)** model, which generates answers strictly grounded in the provided context.

The system is exposed via a **FastAPI** endpoint, returning both the final answer and the retrieved context along with similarity scores for transparency.


## Setup Instructions


### 1. Fork and Clone the repository
```
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create and activate a virtual environment (Windows – Git Bash)
```
python -m venv venv
source venv/Scripts/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set environment variables
```
export PINECONE_API_KEY=your_pinecone_api_key
export GROQ_API_KEY=your_groq_api_key
```

### 5. Ingest and index the document (run once)
```
python script/ingest.py
python script/index_to_pinecone.py
```

### 6. Start the API server
```
uvicorn script.api:app --reload
```

## Open Swagger UI at:
### http://127.0.0.1:8000/docs


## Sample Queries

Use the following sample questions to test the RAG chatbot:

1. What is Agentic AI?  
2. How do agentic systems differ from traditional AI pipelines?  
3. What role do multi-agent systems play in this document?  
4. What challenges are associated with Agentic AI?  
5. What solutions does the document propose for autonomous orchestration?  
6. Explain the concept of autonomous decision-making discussed in the document.



