from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

app = FastAPI(title="Tata RAG Query API")

# Configuration
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load the vector store
try:
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"Error loading vector store: {e}")
    vectorstore = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/query")
def query_documents(request: QueryRequest):
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="Vector store not loaded")
    
    try:
        results = vectorstore.similarity_search(request.query, k=request.top_k)
        response = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
        return {"results": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Tata RAG API is running. Use POST /query to search documents."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)