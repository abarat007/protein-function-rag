import faiss
import numpy as np
import json

# paths
embedding_path = "/Users/abhinavbarat/Documents/bionemo_protein_rag/embeddings/Q13286_embedding.npy"
index_path = "/Users/abhinavbarat/Documents/bionemo_protein_rag/retriever/faiss_index.index"
doc_ids_path = "/Users/abhinavbarat/Documents/bionemo_protein_rag/retriever/doc_ids.npy"
uniprot_texts_path = "/Users/abhinavbarat/Documents/bionemo_protein_rag/data/uniprot_chunks.jsonl"
top_k = 3

index = faiss.read_index(index_path)
doc_ids = np.load(doc_ids_path)
print("Loaded FAISS index and doc IDs.")

doc_db = {}
with open(uniprot_texts_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        doc_db[entry["doc_id"]] = entry["text"]


print("Loading protein embedding:")
query = np.load(embedding_path).astype(np.float32)
query = np.expand_dims(query, axis = 0)

print(f"Searching top {top_k} similar documents...")
distances, indices = index.search(query, top_k)

print("\n Top Matches:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    doc_id = doc_ids[idx]
    text = doc_db.get(doc_id, "[Text not found]")
    print(f"\n#{rank + 1}: {doc_id}")
    print(f"Distance: {dist:.4f}")
    print(f"Text: {text}")
