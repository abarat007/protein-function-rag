import os
import numpy as np
import json
from Bio import SeqIO
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Paths
fasta_path = "/Users/abhinavbarat/Documents/bionemo_protein_rag/data/proteins.fasta"
ground_truth_path = "/Users/abhinavbarat/Documents/bionemo_protein_rag/data/ground_truth.csv"
output_dir = "/Users/abhinavbarat/Documents/bionemo_protein_rag/outputs"
os.makedirs(output_dir, exist_ok=True)

# Load models
biobert_model = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
biobert = SentenceTransformer(biobert_model)
biogpt_model = "microsoft/BioGPT-Large"
generator = pipeline("text-generation", model = biogpt_model, tokenizer = AutoTokenizer.from_pretrained(biogpt_model), device = -1)

# Load ground truth
ground_truth_df = pd.read_csv(ground_truth_path)

# Load retrieved texts (for now, static)
retrieved_texts = [
    "Functions in cell cycle checkpoint regulation and interacts with BRCA1.",
    "This protein is involved in the DNA damage response and double-strand break repair pathways.",
    "Plays a role in mitochondrial fission and may regulate apoptosis."
]

results = []

for record in SeqIO.parse(fasta_path, "fasta"):
    protein_id = record.id.split("|")[1] if "|" in record.id else record.id
    protein_seq = str(record.seq)
    print(f"\nProcessing {protein_id}")

    # Embed (optional save, skipping here)
    embedding = biobert.encode([protein_seq])[0]

    # Build prompt (shortened seq to avoid BioGPT overflow)
    prompt = f"""You are a biomedical research assistant.

Task: Predict the likely biological function of the following protein, using its sequence and related literature.

Protein sequence:
{protein_seq[:300]}...

Top related literature:
1. {retrieved_texts[0]}
2. {retrieved_texts[1]}
3. {retrieved_texts[2]}

Predicted function:"""

    # Run BioGPT
    output = generator(prompt, max_new_tokens = 80, do_sample = True, temperature = 0.7)
    generated_text = output[0]["generated_text"]

    # Clean prediction
    if "Predicted function:" in generated_text:
        predicted_function = generated_text.split("Predicted function:")[-1].strip()
    else:
        predicted_function = generated_text.strip()

    for marker in ["< / FREETEXT >", "< / PARAGRAPH >", "▃", "\n"]:
        if marker in predicted_function:
            predicted_function = predicted_function.split(marker)[0].strip()

    predicted_function = predicted_function.split(".")[0].strip() + "."

    # Get ground truth
    row = ground_truth_df[ground_truth_df["protein_id"] == protein_id]
    if row.empty:
        print(f"⚠️ No ground truth found for {protein_id}, skipping.")
        continue
    true_function = row["function_description"].values[0]

    # Compute similarity
    pred_emb = biobert.encode(predicted_function, convert_to_tensor=True)
    true_emb = biobert.encode(true_function, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(pred_emb, true_emb).item()

    print(f"Predicted Function: {predicted_function}")
    print(f"Ground Truth: {true_function}")
    print(f"Cosine Similarity: {similarity_score:.4f}")

    # Save individual prediction
    with open(os.path.join(output_dir, f"{protein_id}_prediction.txt"), "w") as f:
        f.write(predicted_function)

    # Store result
    results.append({
        "protein_id": protein_id,
        "predicted_function": predicted_function,
        "true_function": true_function,
        "similarity": similarity_score
    })

# Save batch results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "batch_results.csv"), index=False)
print("\nSaved batch results to outputs/batch_results.csv")

# plot summary
plt.figure(figsize=(10, 5))
plt.bar(results_df["protein_id"], results_df["similarity"], color="skyblue")
plt.ylim(0, 1)
plt.ylabel("Cosine Similarity")
plt.title("Predicted vs Ground Truth Similarity (All Proteins)")
plt.xticks(rotation = 45, ha = "right")
plt.tight_layout()
plt.show()