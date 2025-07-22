# 🧬 Protein Function RAG Explorer 🧬

A Streamlit app that predicts the biological function of proteins using a retrieval-augmented generation (RAG) pipeline.  
Given a protein FASTA sequence, the application:

✅ Embeds it using BioBERT  
✅ Retrieves top relevant literature with FAISS  
✅ Generates function predictions with BioGPT  
✅ Displays an interactive 3D protein model from RCSB PDB

---

## 🌟 Features

- Batch evaluation on curated ground-truth protein dataset
- Live FASTA input and prediction
- Cosine similarity comparison to ground-truth functions
- Embedded 3D protein structure viewer (from RCSB PDB)
- Dockerized setup for easy deployment

---

## 📦 Project Structure

```
/app/
  └── streamlit_app.py     # Main Streamlit app
/data/
  └── ground_truth.csv
  └── uniprot_chunks.jsonl
/retriever/
  └── faiss_index.index
  └── doc_ids.npy
/outputs/
  └── batch_results.csv
requirements.txt
Dockerfile
.dockerignore
README.md
```

---

## 🚀 Install (dev mode)

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 🐳 Run via Docker

```bash
docker build -t proteinrag-app .
docker run -p 8501:8501 proteinrag-app
```

Then open: http://localhost:8501 in your browser.

---

## 💡 Usage

1️⃣ Select a protein from the sidebar dropdown to view batch predictions  
2️⃣ Paste a new protein FASTA in the textarea to run live prediction  
3️⃣ View the predicted function, retrieved documents, and 3D model  
4️⃣ Download batch results or live predictions

---

## 🌐 Dependencies

- [transformers](https://huggingface.co/docs/transformers/)
- [sentence-transformers](https://www.sbert.net/)
- [faiss](https://faiss.ai/)
- [streamlit](https://streamlit.io/)
- [pandas, numpy, matplotlib]

---

## 🛠️ Development Tips

- Make sure `outputs/` and `retriever/` folders exist before running.
- Docker image pulls models at runtime if not cached; ensure enough disk space.

---

## 📸 Demo Screenshot

![App Screenshot](./demo_screenshot.png)

---

## 📄 License

MIT License

