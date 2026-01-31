from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("data/ebook.pdf")


def load_pdf_text(pdf_path: Path) -> str:
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def chunk_text(text: str):
    """Splits text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_text(text)


if __name__ == "__main__":
    print(" Loading PDF...")
    raw_text = load_pdf_text(PDF_PATH)

    print(f" Extracted {len(raw_text)} characters")

    print(" Chunking text...")
    chunks = chunk_text(raw_text)

    print(f" Created {len(chunks)} chunks")

    print("\n🔍 Sample chunk:\n")
    print(chunks[0][:500])
