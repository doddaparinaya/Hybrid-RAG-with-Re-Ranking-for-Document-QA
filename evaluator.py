# evaluator.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def quality_of_retrieval(embedding_model, query, docs, threshold=0.3):
    q_emb = embedding_model.embed_query(query)
    d_embs = [embedding_model.embed_query(doc.page_content) for doc in docs]

    sims = cosine_similarity([q_emb], d_embs)[0]
    score = np.mean(sims)

    return ("GOOD" if score > threshold else "LOW", score)