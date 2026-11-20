"""
Metode similarity menggunakan Cosine Similarity (vektor TF-IDF).
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_scores(query_vec, doc_matrix) -> np.ndarray:
    """
    query_vec: sparse matrix (1 x n_features) hasil vectorizer.transform()
    doc_matrix: sparse matrix (n_docs x n_features)
    return: array skor (n_docs,)
    """
    return cosine_similarity(query_vec, doc_matrix).flatten()