import pandas as pd
import numpy as np

from clean import clean_judgements
from embeddings import get_embedding
from semantcigraph import find_similar_cases, build_graph

def main():
    df = pd.read_csv("court_data.csv")
    
    # 2.Clean
    df['judge_clean'] = df['judgement'].apply(clean_judgements)
    
    # 3.embeddings 
    df['embedding'] = df['judge_clean'].apply(lambda x: get_embedding(x))
    
    # 4.embeddings to numpy array for similarity calculations
    embeddings = np.array(df['embedding'].tolist())
    
    # 5.pick case index
    user_case_index = 0
    
    # 6.similar cases
    similar_indices = find_similar_cases(embeddings, user_case_index, top_n=5)
    all_indices = [user_case_index] + list(similar_indices)
    
    # 7.semantic graph 
    graph = build_graph(df['judge_clean'].tolist(), embeddings, all_indices, threshold=0.5)
    
    # 8.visualize the graph
    
    # Example: print nodes and edges
    print("Graph nodes:", graph.nodes(data=True))
    print("Graph edges:", graph.edges(data=True))

if __name__ == "__main__":
    main()
