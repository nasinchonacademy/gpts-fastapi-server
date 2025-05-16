
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_and_chunk(directory, chunk_size=500):
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)
    return chunks

def main():
    data_dir = "data/extracted"
    index_dir = "index"
    os.makedirs(index_dir, exist_ok=True)

    chunks = load_and_chunk(data_dir)
    model = SentenceTransformer("hkunlp/instructor-xl")
    embeddings = model.encode(chunks, batch_size=16, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, os.path.join(index_dir, "vec.index"))

    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print("Index and chunks saved.")

if __name__ == "__main__":
    main()
