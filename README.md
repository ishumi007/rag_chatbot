# 📚 RAG-based AI Chatbot (LangGraph + Pinecone + Groq)

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that answers questions **strictly from a given PDF document**.

It uses:
- **SentenceTransformers** for free, local embeddings
- **Pinecone** as a vector database
- **LangGraph** to orchestrate the RAG workflow
- **Groq (LLaMA 3.1)** as a free hosted LLM
- **FastAPI** to expose a production-ready Chat API

---

## 🚀 Features

- 📄 PDF ingestion and chunking
- 🧠 Semantic search using vector embeddings
- 🔍 Transparent retrieval (shows retrieved chunks + similarity scores)
- 🤖 Hallucination-safe answers (context-only)
- 🌐 REST API with Swagger UI
- 💻 Memory-efficient (runs on CPU, no local LLM)

---

## 🧠 Architecture Overview

User Question
↓
SentenceTransformer (query embedding)
↓
Pinecone Vector DB (Top-K retrieval)
↓
LangGraph Workflow
├── Retrieve Node
└── Generate Node (Groq LLaMA 3.1)
↓
Answer + Retrieved Context + Confidence Scores


### Why LangGraph?
LangGraph allows explicit control over:
- Retrieval
- Generation
- State passing
- Future extensions (retries, relevance checks, multi-agent flows)

---

## 🛠️ Setup Instructions

### 1️⃣ fork then Clone the repository


2️⃣ Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set environment variables
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key

🧱 Step-by-Step Execution
🔹 Step 1: Ingest & chunk the PDF
python script/ingest.py

🔹 Step 2: Create embeddings & index into Pinecone (run once)
python script/index_to_pinecone.py

🔹 Step 3: Run the API server
uvicorn script.api:app --reload

Open:
http://127.0.0.1:8000/docs

🔌 API Usage
POST /chat

Request

{
  "question": "What is agentic AI?"
}


Response
{
  "answer": "...",
  "contexts": [
    {
      "text": "Agentic AI refers to...",
      "score": 0.84
    }
  ]
}


🧪 Sample Queries

Try these in Swagger UI or via curl:

1. What is Agentic AI?
2. How do agentic systems differ from traditional AI pipelines?
3. What role do multi-agent systems play in this document?
4. What challenges are associated with agentic AI?
5. What solutions does the document propose for autonomous orchestration?
6. Explain the concept of autonomous decision-making discussed here

🔍 Confidence Scores Explained

Scores are cosine similarity values from Pinecone
Range: 0.0 → 1.0
Higher score = higher semantic relevance

| Score Range   | Interpretation    |
| ------------- | ----------------- |
| `0.80+`       | Very strong match |
| `0.65 – 0.80` | Relevant          |
| `< 0.60`      | Weak / noisy      |

🔐 Hallucination Prevention

The LLM is explicitly instructed to:

* Answer only from retrieved context
* Respond with “I don’t know based on the document” if information is missing


| Layer         | Technology                    |
| ------------- | ----------------------------- |
| Embeddings    | SentenceTransformers (MiniLM) |
| Vector DB     | Pinecone                      |
| Orchestration | LangGraph                     |
| LLM           | Groq (LLaMA 3.1)              |
| API           | FastAPI                       |
| Language      | Python                        |
