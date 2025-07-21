import streamlit as st
import pandas as pd
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer

st.set_page_config(page_title = "Protein Function RAG", layout = "wide")

@st.cache_resource
def load_biobert():
    return SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

@st.cache_resource
def load_biogpt():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
    return pipeline("text-generation", model="microsoft/BioGPT", tokenizer = tokenizer, device = -1)

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("./retriever/faiss_index.index")
    doc_ids = np.load("./retriever/doc_ids.npy")
    with open("./data/uniprot_chunks.jsonl", "r") as f:
        doc_db = {json.loads(line)["doc_id"]: json.loads(line)["text"] for line in f}
    return index, doc_ids, doc_db

@st.cache_data
def load_results():
    return pd.read_csv("./outputs/batch_results.csv")

biobert = load_biobert()
biogpt = load_biogpt()
index, doc_ids, doc_db = load_faiss_index()
results_df = load_results()

st.sidebar.title("Protein Selector")
protein_ids = results_df["protein_id"].tolist()
selected_protein = st.sidebar.selectbox("Choose a protein:", protein_ids)

row = results_df[results_df["protein_id"] == selected_protein].iloc[0]

st.title("🧬 Protein Function RAG Explorer 🧬")
st.markdown(f"### Selected Protein: `{selected_protein}`")
st.markdown(f"**Predicted Function:** {row['predicted_function']}")
st.markdown(f"**Ground Truth Function:** {row['true_function']}")
st.markdown(f"**Cosine Similarity:** {row['similarity']:.4f}")

# Embed 3D viewer for selected protein
st.markdown("### 🧬 3D Structure (RCSB PDB) for Selected Protein 🧬")
st.components.v1.iframe(f"https://www.rcsb.org/groups/3d-sequence/polymer_entity/{selected_protein}", height = 600, width = 1100)

st.download_button("Download Batch Results CSV", data=results_df.to_csv(index=False), file_name = "batch_results.csv")

st.markdown("---")

st.markdown("## Paste New Protein FASTA for Live Prediction")
fasta_input = st.text_area("Paste FASTA here (include header line `>`)")

if st.button("Run Live Prediction") and fasta_input:
    with st.spinner("Running live prediction..."):
        fasta_lines = fasta_input.strip().split("\n")
        header_line = [line for line in fasta_lines if line.startswith(">")][0]
        uniprot_id = header_line.split("|")[1] if "|" in header_line else "UNKNOWN"
        protein_seq = "".join(line.strip() for line in fasta_lines if not line.startswith(">"))

        query_emb = biobert.encode(protein_seq, convert_to_tensor=False).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_emb, 3)
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
        if "Predicted function:" in generated_text:
            predicted_function = generated_text.split("Predicted function:")[-1].strip()
        else:
            predicted_function = generated_text.strip()

        predicted_function = predicted_function.split(".")[0].strip() + "."

        st.markdown(f"### 🧪 Predicted Function 🧪\n{predicted_function}")
        st.markdown("### 🔍 Top Retrieved Documents 🔍")
        for i, text in enumerate(retrieved_texts, 1):
            st.markdown(f"{i}. {text}")

        # Embed 3D RCSB viewer for pasted protein
        st.markdown("### 🧬 3D Structure (RCSB PDB) for Pasted Protein")
        st.components.v1.iframe(f"https://www.rcsb.org/groups/3d-sequence/polymer_entity/{uniprot_id}", height = 600, width = 1100)

        st.download_button("Download Live Prediction", data = predicted_function, file_name = "live_prediction.txt")
