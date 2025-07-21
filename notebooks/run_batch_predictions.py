import os
import numpy as np
import pandas as pd
import faiss
import json
from Bio import SeqIO
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

gt_df = pd.read_csv("/Users/abhinavbarat/Documents/bionemo_protein_rag/data/ground_truth.csv")

index = faiss.read_index("/Users/abhinavbarat/Documents/bionemo_protein_rag/retriever/faiss_index.index")
doc_ids = np.load("/Users/abhinavbarat/Documents/bionemo_protein_rag/retriever/doc_ids.npy")
with open("/Users/abhinavbarat/Documents/bionemo_protein_rag/data/uniprot_chunks.jsonl", "r") as f:
    doc_db = {json.loads(line)["doce_id"]: json.loads(line)["text"] for line in f}

biogpt = pipeline("text-generation", model = "microsoft/BioGPT-Large", device = -1)
biobert = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

results = []

for record in SeqIO.parse("/Users/abhinavbarat/Documents/bionemo_protein_rag/data/proteins.fasta", "fasta"):
    protein_id = record.id.split('|')[1] if '|' in record.id else record.id
    protein_seq = str(record.seq)
    print(f"Processing {protein_id}")

    protein_emb = biobert.encode(protein_seq, convert_to_tensor=False).astype('float32').reshape(1,-1)

    top_k = 3
    distances, indices = index.search(protein_emb, top_k)
    retrieved_texts = [doc_db.get(doc_ids[idx], "[Text not found]") for idx in indices[0]]

    prompt = f"""You are a biomedical research assistant.

Task: Predict the likely biological function of the following protein, using its sequence and related literature.

Protein sequence:
{protein_seq[:300]}...

Top related literature:
1. {retrieved_texts[0]}
2. {retrieved_texts[1]}
3. {retrieved_texts[2]}

Predicted function:"""
    
    output = biogpt(prompt, max_new_tokens = 80, do_sample = True, temperature = 0.7)
    generated_text = output[0]["generated_text"]
    predicted_function = generated_text.split("Predicted function:")[-1].strip().split(".")[0] + "."

    row = gt_df[gt_df["protein_id"] == protein_id]
    if row.empty:
        print(f"Skipping {protein_id}, no ground truth.")
        continue
    true_function = row["function_description"].values[0]

    pred_emb = biobert.encode(predicted_function, convert_to_tensor=True)
    true_emb = biobert.encode(true_function, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(pred_emb, true_emb).item()

    results.append({"protein_id": protein_id, "predicted_function": predicted_function, "true_function": true_function, "similarity": similarity})

os.makedirs("outputs", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/batch_results.csv", index = False)
print("Saved batch results to outputs/batch_results.csv")