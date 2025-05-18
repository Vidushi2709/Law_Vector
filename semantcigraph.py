from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def find_similar_cases(embeddings, case_index, top_n=3):
    similarities = cosine_similarity([embeddings[case_index]], embeddings)[0]
    similar_indices = similarities.argsort()[::-1][1:top_n+1]  # exclude itself
    return similar_indices

def build_graph(texts, embeddings, indices, threshold=0.5):
    g = nx.Graph()
    for i in indices:
        g.add_node(i, text=texts[i])
    subset_embeddings = embeddings[indices]
    sims = cosine_similarity(subset_embeddings)
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            if sims[i][j] > threshold:
                g.add_edge(indices[i], indices[j], weight=sims[i][j])
    return g

