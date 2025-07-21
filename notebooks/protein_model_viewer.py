import requests
import pandas as pd

def fetch_protein_model(uniprot_id):
    url = f"https://www.rcsb.org/groups/3d-sequence/polymer_entity/{uniprot_id}"
    response = requests.get(url)

if __name__ == "__name__":
    protein_id = input("Enter UniProt Protein ID:").strip()
    