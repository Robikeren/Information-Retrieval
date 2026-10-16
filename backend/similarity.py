"""
4 metode similarity untuk perbandingan:
1. Cosine Similarity      (vektor TF-IDF)
2. Jaccard Coefficient    (himpunan biner)
3. Sørensen-Dice          (himpunan biner)
4. Overlap Similarity     (himpunan biner)
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ============================================================
# COSINE — pakai matriks TF-IDF
# ============================================================
def cosine_scores(query_vec, doc_matrix) -> np.ndarray:
    """
    query_vec: sparse matrix (1 x n_features) hasil vectorizer.transform()
    doc_matrix: sparse matrix (n_docs x n_features)
    return: array skor (n_docs,)
    """
    return cosine_similarity(query_vec, doc_matrix).flatten()


# ============================================================
# METODE BERBASIS HIMPUNAN
# ============================================================
def jaccard_similarity(set_q: set, set_d: set) -> float:
    if not set_q or not set_d:
        return 0.0
    inter = len(set_q & set_d)
    union = len(set_q | set_d)
    return inter / union if union > 0 else 0.0


def dice_similarity(set_q: set, set_d: set) -> float:
    if not set_q or not set_d:
        return 0.0
    inter = len(set_q & set_d)
    return (2 * inter) / (len(set_q) + len(set_d))


def overlap_similarity(set_q: set, set_d: set) -> float:
    if not set_q or not set_d:
        return 0.0
    inter = len(set_q & set_d)
    return inter / min(len(set_q), len(set_d))


# ============================================================
# WRAPPER: hitung skor semua dokumen untuk satu query
# ============================================================
def compute_all_scores(
    query_tokens: list[str],
    doc_tokens_list: list[list[str]],
    query_vec=None,
    doc_matrix=None,
) -> dict:
    """
    Return dict: { 'cosine': np.array, 'jaccard': np.array,
                   'dice': np.array, 'overlap': np.array }
    """
    set_q = set(query_tokens)
    n = len(doc_tokens_list)

    jaccard = np.zeros(n)
    dice = np.zeros(n)
    overlap = np.zeros(n)

    for i, doc_tokens in enumerate(doc_tokens_list):
        set_d = set(doc_tokens)
        jaccard[i] = jaccard_similarity(set_q, set_d)
        dice[i] = dice_similarity(set_q, set_d)
        overlap[i] = overlap_similarity(set_q, set_d)

    result = {
        "jaccard": jaccard,
        "dice": dice,
        "overlap": overlap,
    }

    if query_vec is not None and doc_matrix is not None:
        result["cosine"] = cosine_scores(query_vec, doc_matrix)

    return result