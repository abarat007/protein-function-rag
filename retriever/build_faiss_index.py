import pandas as pd
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

df = pd.read_csv("/Users/abhinavbarat/Documents/bionemo_protein_rag/data/ground_truth.csv")

biobert = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

embeddings = []
doc_ids = []
jsonl_lines = []

for idx, row in df.iterrows():
    protein_id = row['protein_id']
    text = row["function_description"]

    if pd.isna(text) or not text.strip():
        continue

    emb = biobert.encode(text)
    embeddings.append(emb)
    doc_ids.append(protein_id)

    jsonl_lines.append(({"doce_id": protein_id, "text": text}))

embeddings = np.vstack(embeddings).astype('float32')

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "/Users/abhinavbarat/Documents/bionemo_protein_rag/retriever/faiss_index.index")
np.save("/Users/abhinavbarat/Documents/bionemo_protein_rag/retriever/doc_ids.npy", np.array(doc_ids))
print("Saved FAISS index and doc IDs")

with open("/Users/abhinavbarat/Documents/bionemo_protein_rag/data/uniprot_chunks.jsonl", "w") as f:
    for entry in jsonl_lines:
        f.write(json.dumps(entry) + "\n")
    print("Saved uniprot_chunks.jsonl")