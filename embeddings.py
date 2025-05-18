import pandas as pd
import google.generativeai as genai
from tqdm import tqdm  
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

data= pd.read_csv("cleaned_judgements.csv")
genai.configure(api_key="AIzaSyDVhtEzhMMNqVWM1nGaA7OotMs-0cw89jA")

'''def get_embeddings(text):
    data= pd.read_csv("cleaned_judgements.csv")
    genai.configure(api_key="AIzaSyDVhtEzhMMNqVWM1nGaA7OotMs-0cw89jA")

    df = pd.read_csv("cleaner.csv")  # assume 'summary' column
    text_column = "judge_clean"
    chunk_size = 10
    MAX_CHARS = 30000

    def truncate(text):
        return text[:MAX_CHARS] if len(text) > MAX_CHARS else text

    all_embeddings = []

    for start in tqdm(range(0, len(df), chunk_size)):
        end = start + chunk_size
        chunk = df[text_column].iloc[start:end]

        for text in chunk:
            try:
                text = str(text) if pd.notna(text) else ""
                text = truncate(text)

                response = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )

                embedding = response["embedding", None]

            except Exception as e:
                print(f"Error at index {start}: {e}")
                embedding = None

            all_embeddings.append(embedding)

    df["embedding"] = all_embeddings
    df.to_csv("cases_with_embeddings.csv", index=False)'''

def get_embedding(text, max_chars=30000):
    text = text[:max_chars]
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]
