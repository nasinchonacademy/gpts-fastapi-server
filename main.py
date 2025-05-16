
from fastapi import FastAPI, Request
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("hkunlp/instructor-xl")
index = faiss.read_index("index/vec.index")

with open("index/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

app = FastAPI()

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    question = data["query"]
    vec = model.encode([question])
    D, I = index.search(np.array(vec).astype("float32"), k=5)
    results = [chunks[i] for i in I[0]]
    return {"results": results}
