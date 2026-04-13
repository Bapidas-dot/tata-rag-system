import os

import fitz  # PyMuPDF (fallback reader)
from transformers import logging as transformers_logging

# LangChain tools for loading and splitting documents
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ChromaDB is our local vector database - no server needed, saves to disk
from langchain_community.vectorstores import Chroma

# Sentence Transformers gives us FREE local embeddings - no API key needed
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Configurations for embeddings, models, chunk size, overlaps
PDF_PATHS = [
    "nexon-adas-brochure.pdf",
    "nexon-ev-brochure.pdf",
]
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 580
CHUNK_OVERLAP = 50


def load_pdf(paths=None):
    """Load one or more PDF files and return Document objects."""
    if paths is None:
        paths = PDF_PATHS
    if isinstance(paths, str):
        paths = [path.strip() for path in paths.split(",") if path.strip()]

    documents = []
    for path in paths:
        try:
            loader = PyPDFLoader(path)
            loaded = list(loader.load())
            print(f"Loaded {len(loaded)} docs from {path} via PyPDFLoader")
            documents.extend(loaded)
            continue
        except Exception as e:
            print(f"PyPDFLoader failed for {path}: {e}")

        # Fallback to fitz when pypdf cannot parse.
        try:
            doc = fitz.open(path)
            page_docs = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_docs.append(
                    Document(page_content=page_text, metadata={"source": path, "page": page_num})
                )
            print(f"Loaded {len(page_docs)} docs from {path} via fitz fallback")
            documents.extend(page_docs)
        except Exception as e:
            print(f"fitz fallback failed for {path}: {e}")

    if not documents:
        raise ValueError(f"No documents loaded from paths: {paths}")

    print(f"OK loaded {len(documents)} documents from {paths}")
    return documents


def split_documents(documents):
    """Split documents into smaller chunks for embedding."""
    print("\nSplitting documents into chunks...")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", ","],
    )

    chunks = splitter.split_documents(documents)
    print(f"OK created {len(chunks)} chunks total")
    return chunks


def build_vectorstore(chunks):
    """Create a Chroma vector store for the provided chunks."""
    transformers_logging.set_verbosity_error()
    print("\nCreating embeddings and saving to ChromaDB...")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    print(f"OK vector store saved to: {CHROMA_DB_PATH}/")
    return vectorstore


if __name__ == "__main__":
    # Load PDFs, split into chunks, and build the vector store
    pdf_docs = load_pdf(
        "/workspaces/tata-rag-system/nexon-adas-brochure.pdf, /workspaces/tata-rag-system/nexon-ev-brochure.pdf"
    )
    chunks = split_documents(pdf_docs)
    vectorstore = build_vectorstore(chunks)

    # Combine both document list into one
    all_documents = pdf_docs
    print(f"\nTotal documents loaded: {len(all_documents)}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print("Vector store created successfully.")
    print(f"Database saved at: {CHROMA_DB_PATH}/")

    print("=" * 50)
    print(" OK Done! You can now run 'python query.py' to ask questions about the PDFs.")
    print(f" Database saved at: {CHROMA_DB_PATH}/")
    print()
    print(f"Next steps: Run the API server with:")
    print("  python api_server.py")
    print("=" * 50)