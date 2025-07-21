import pandas as pd
import requests

def fetch_protein_info(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch data for {uniprot_id}")
        return None

    data = response.json()

    # Extract GO terms
    go_terms = []
    if "uniProtKBCrossReferences" in data:
        for ref in data["uniProtKBCrossReferences"]:
            if ref["database"] == "GO":
                go_id = ref["id"]
                go_terms.append(go_id)
    go_term_str = "|".join(go_terms)

    # Extract function description
    function_desc = ""
    if "comments" in data:
        for comment in data["comments"]:
            if comment["commentType"] == "FUNCTION":
                texts = [text["value"] for text in comment.get("texts", [])]
                if texts:
                    function_desc = texts[0]
                    break

    return {
        "protein_id": uniprot_id,
        "function_description": function_desc,
        "go_terms": go_term_str
    }

def update_ground_truth_csv(protein_info, csv_path):
    df = pd.read_csv(csv_path)

    if protein_info["protein_id"] in df["protein_id"].values:
        df.loc[df["protein_id"] == protein_info["protein_id"], "function_description"] = protein_info["function_description"]
        df.loc[df["protein_id"] == protein_info["protein_id"], "go_terms"] = protein_info["go_terms"]
        print(f"Updated ground_truth.csv for {protein_info['protein_id']}")
    else:
        new_row = pd.DataFrame({
            "protein_id": [protein_info["protein_id"]],
            "function_description": [protein_info["function_description"]],
            "go_terms": [protein_info["go_terms"]]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        print(f"Added new row in ground_truth.csv for {protein_info['protein_id']}")

    df.to_csv(csv_path, index=False)
    print(f"Saved updated ground_truth.csv")

def update_proteins_fasta(uniprot_id, fasta_path):
    # Check if already exists
    exists = False
    with open(fasta_path, "r") as f:
        for line in f:
            if line.strip().startswith(f">{uniprot_id}") or f"|{uniprot_id}|" in line:
                exists = True
                break

    if exists:
        print(f"⚠️ Protein {uniprot_id} already exists in proteins.fasta, skipping append")
        return

    # Fetch full FASTA from UniProt
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch FASTA for {uniprot_id}")
        return

    fasta_text = response.text

    # Append to local proteins.fasta
    with open(fasta_path, "a") as f:
        f.write(fasta_text.strip() + "\n")

    print(f"Added {uniprot_id} to proteins.fasta with full FASTA header")

if __name__ == "__main__":
    protein_id = input("Enter UniProt protein ID: ").strip()
    csv_path = "/Users/abhinavbarat/Documents/bionemo_protein_rag/data/ground_truth.csv"
    fasta_path = "/Users/abhinavbarat/Documents/bionemo_protein_rag/data/proteins.fasta"

    protein_info = fetch_protein_info(protein_id)
    if protein_info:
        update_ground_truth_csv(protein_info, csv_path)
        update_proteins_fasta(protein_id, fasta_path)
    else:
        print("⚠️ Could not retrieve protein info; skipping update.")
