import faiss 
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
import os


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

#give the results to perplexity api for bot function

def ask_about_cases(user_ask, law_cases):
    api_key= os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        raise EnvironmentError("PERPLEXITY_API_KEY is not set in environment variables.")

    contxt= "\n\n".join([
        f"CASE NAME: {case['case_name']}\n JUDGEMENT: {case['judgement']}"
        for case in law_cases
    ])

    message=[
        {"role": "system", "content": ""},
        {"role": "user", "content": f"{contxt}\n\nUser's Question: {user_ask}"}
    ]

    header= {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar-pro",
        "messages": message,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    
    response = requests.post("https://api.perplexity.ai/chat/completions", headers=header, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]