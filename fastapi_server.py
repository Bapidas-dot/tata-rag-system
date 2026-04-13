from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import tempfile

app = FastAPI(title="Tata RAG System")

# Configuration
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Global vector store
VECTOR_DB = None

class QueryRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    global VECTOR_DB

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        # Parse PDF (assuming parse_pdf is available, but for now, use simple loader)
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split documents
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=580, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Create vector store
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        VECTOR_DB = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
        )

        return {
            "message": "Ingested successfully",
            "chunks": len(chunks)
        }
    finally:
        os.unlink(temp_file_path)

@app.post("/query")
def query(request: QueryRequest):
    if VECTOR_DB is None:
        return {"error": "No documents ingested yet"}

    question = request.question or request.query
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty. Use JSON body { \"question\": \"...\" }.")

    docs = VECTOR_DB.similarity_search(question, k=5)
    context = "\n".join([d.page_content for d in docs])

    # For now, return context; in full implementation, use LLM to generate answer
    return {
        "answer": context,  # Placeholder
        "sources": len(docs)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




    