import faiss 
import pandas as pd
from sentence_transformers import SentenceTransformer


#index and meta data
index = faiss.read_index("law_index.faiss")
df = pd.read_csv("law_metadata.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_law_cases(query, k=5):
    query = query.lower()
    query_embedding = model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k)

    #no matches found
    if I.shape[1] == 0 or I[0][0] == -1:
        return []  

    results = []
    for idx in I[0]:
        result = {
            "case_name": df.iloc[idx]['case_name'],
            "case_id": df.iloc[idx]['case_id'],
            "judgement": df.iloc[idx]['judgement'][:300] + "..."
        }
        results.append(result)
    return results