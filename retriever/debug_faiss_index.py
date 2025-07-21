import faiss
import numpy as np

# load index and doc ids
index = faiss.read_index("/Users/abhinavbarat/Documents/bionemo_protein_rag/retriever/faiss_index.index")
doc_ids = np.load("/Users/abhinavbarat/Documents/bionemo_protein_rag/retriever/doc_ids.npy")

print(f"FAISS index contains {index.ntotal} vectors.")
print(f"Corresponding doc IDS: {doc_ids}")
