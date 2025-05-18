import pandas as pd
import spacy
import string
from tqdm import tqdm

def clean_judgements(df, text_column='judgement', cleaned_column='judge_clean', batch_size=32):
    df[text_column] = df[text_column].astype(str).str.lower()
    
    def remove_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation)).strip() if isinstance(text, str) else text
    
    df[text_column] = df[text_column].apply(remove_punc)
    
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    
    def remove_stopwords_spacy(doc):
        return " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
    
    df[text_column] = df[text_column].fillna("")
    texts = df[text_column].tolist()
    
    cleaned_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Cleaning judgements"):
        batch_texts = texts[i:i+batch_size]
        docs = nlp.pipe(batch_texts)
        for doc in docs:
            cleaned_texts.append(remove_stopwords_spacy(doc))
    
    df[cleaned_column] = cleaned_texts
    return df
