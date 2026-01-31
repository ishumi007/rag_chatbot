from pathlib import Path
from typing import List

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from ingest import load_pdf_text, chunk_text


# ================== CONFIG ==================
INDEX_NAME = "rag-chatbot-index-v2"   # NEW index name
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
# ============================================


# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR.parent / "data" / "ebook.pdf"


# ---------- CLIENTS ----------
print("🔌 Loading embedding model (CPU, free)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

pc = Pinecone()   # API key from env


# ---------- CREATE INDEX ----------
def create_index_if_not_exists():
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print("🧠 Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
    else:
        print("✅ Pinecone index already exists")


# ---------- EMBEDDINGS ----------
def embed_texts(texts: List[str]):
    return embedding_model.encode(
        texts,
        show_progress_bar=True
    ).tolist()


# ---------- MAIN ----------
if __name__ == "__main__":
    print("📄 Loading & chunking PDF...")
    text = load_pdf_text(PDF_PATH)
    chunks = chunk_text(text)

    print(f"✅ {len(chunks)} chunks ready")

    create_index_if_not_exists()
    index = pc.Index(INDEX_NAME)

    print("🔢 Generating embeddings (FREE)...")
    embeddings = embed_texts(chunks)

    print("📤 Uploading vectors to Pinecone...")
    vectors = []

    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        vectors.append(
            {
                "id": f"chunk-{i}",
                "values": vector,
                "metadata": {
                    "text": chunk
                }
            }
        )

    index.upsert(vectors=vectors)

    print("🎉 STEP 2 COMPLETE — FREE & CONSISTENT")
