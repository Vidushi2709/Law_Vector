import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

CSV_FILE = "law_metadata.csv"
FAISS_FILE = "law_index.faiss"

model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv(CSV_FILE)
index = faiss.read_index(FAISS_FILE)

def add_case_to_db(case_name, case_id, judgement_text):
    global df, index

    #encode it
    embedding = model.encode([judgement_text]).astype("float32")

    #embedding for faiss
    index.add(embedding)

    #add to csv
    new_row = {"case_name": case_name, "case_id": case_id, "judgement": judgement_text}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    #save 
    df.to_csv(CSV_FILE, index=False)
    faiss.write_index(index, FAISS_FILE)

    return "Case added successfully!"
