from fastapi import FastAPI

app = FastAPI(title="Multimodal RAG - Nixon Brochure")

@app.get("/")
def root():
    return {"status": "RAG System Running"}
