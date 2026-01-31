from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# ================= CONFIG =================
INDEX_NAME = "rag-chatbot-index-v2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
LLM_MODEL = "llama-3.1-8b-instant"
# =========================================


# ---------- CLIENTS ----------
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
pc = Pinecone()
index = pc.Index(INDEX_NAME)
llm = Groq()


# ---------- FASTAPI APP ----------
app = FastAPI(title="RAG Chat API", version="1.0")


# ---------- REQUEST / RESPONSE MODELS ----------
class ChatRequest(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    contexts: List[RetrievedChunk]


# ---------- GRAPH STATE ----------
class GraphState(dict):
    question: str
    context: List[Dict]
    answer: str


# ---------- RETRIEVER NODE ----------
def retrieve(state: GraphState) -> GraphState:
    query_embedding = embedding_model.encode(
        state["question"]
    ).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    retrieved = [
        {
            "text": match["metadata"]["text"],
            "score": round(match["score"], 4)
        }
        for match in results["matches"]
    ]

    return {
        "question": state["question"],
        "context": retrieved,
        "answer": ""
    }


# ---------- GENERATOR NODE ----------
def generate(state: GraphState) -> GraphState:
    context_text = "\n\n".join(
        chunk["text"] for chunk in state["context"]
    )

    prompt = f"""
You are a question-answering AI.

RULES:
- Answer ONLY using the context below.
- If the answer is not present, say "I don't know based on the document."
- Do NOT use outside knowledge.

CONTEXT:
{context_text}

QUESTION:
{state["question"]}
"""

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "question": state["question"],
        "context": state["context"],
        "answer": response.choices[0].message.content
    }


# ---------- BUILD GRAPH ----------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_app = build_graph()


# ---------- API ENDPOINT ----------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = rag_app.invoke({"question": request.question})

    return {
        "answer": result["answer"],
        "contexts": result["context"]
    }


@app.get("/")
def root():
    return {
        "message": "RAG Chat API is running. Go to /docs to use it."
    }
