from typing import TypedDict, List, Dict

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
print(" Initializing clients...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
pc = Pinecone()
index = pc.Index(INDEX_NAME)
llm = Groq()


# ---------- GRAPH STATE ----------
class RetrievedChunk(TypedDict):
    text: str
    score: float


class GraphState(TypedDict):
    question: str
    context: List[RetrievedChunk]
    answer: str


# ---------- RETRIEVER NODE ----------
def retrieve(state: GraphState) -> GraphState:
    print(" Retrieving relevant chunks...")

    query_embedding = embedding_model.encode(
        state["question"]
    ).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    retrieved_chunks: List[RetrievedChunk] = []

    for match in results["matches"]:
        retrieved_chunks.append(
            {
                "text": match["metadata"]["text"],
                "score": round(match["score"], 4)
            }
        )

    return {
        "question": state["question"],
        "context": retrieved_chunks,
        "answer": ""
    }


# ---------- GENERATOR NODE ----------
def generate(state: GraphState) -> GraphState:
    print("🧠 Generating answer...")

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
        messages=[
            {"role": "user", "content": prompt}
        ],
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


# ---------- CLI CHAT ----------
if __name__ == "__main__":
    app = build_graph()

    print("\n RAG Chatbot is ready! (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() in {"exit", "quit"}:
            break

        result = app.invoke({"question": question})

        print("\n Answer:\n")
        print(result["answer"])

        print("\nRetrieved Context Chunks:")
        for i, chunk in enumerate(result["context"], start=1):
            print(f"\n--- Chunk {i} (score: {chunk['score']}) ---")
            print(chunk["text"][:500], "..." if len(chunk["text"]) > 500 else "")

        print("\n" + "=" * 60)
