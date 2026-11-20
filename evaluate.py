"""
Script evaluasi untuk Cosine Similarity.
Jalankan: python evaluate.py
"""
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.preprocessing import preprocess_to_tokens
from backend.similarity import cosine_scores
from backend.evaluator import evaluate_single_method

K = 10
ARTIFACT_DIR = "artifacts"
GT_PATH      = "ground_truth/labels.json"
RESULT_DIR   = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 60)
    print(f"EVALUASI COSINE SIMILARITY @K={K}")
    print("=" * 60)

    # Load artifacts
    with open(f"{ARTIFACT_DIR}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/processed_docs.pkl", "rb") as f:
        docs = pickle.load(f)
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    doc_ids = docs["doc_ids"]
    ground_truth = {q: set(v["relevant_doc_ids"]) for q, v in gt_raw.items()}
    queries = list(ground_truth.keys())

    print(f"Dokumen   : {len(doc_ids)}")
    print(f"Query uji : {len(queries)}")

    rankings = {}
    print(f"\nMenghitung skor Cosine untuk {len(queries)} query...")

    for qi, query in enumerate(queries, 1):
        q_tokens = preprocess_to_tokens(query)
        q_vec    = vectorizer.transform([" ".join(q_tokens)])
        
        scores = cosine_scores(q_vec, tfidf_matrix)
        ranked_idx = np.argsort(scores)[::-1]
        
        rankings[query] = [int(doc_ids[i]) for i in ranked_idx]
        print(f"  [{qi}/{len(queries)}] '{query[:40]}...' OK")

    # Evaluasi
    print(f"\nEvaluasi Cosine Similarity @K={K}...")
    res = evaluate_single_method(rankings, ground_truth, k=K)

    print("\n" + "=" * 60)
    print(f"Precision@{K} : {res['precision@k']:.4f}")
    print(f"Recall@{K}    : {res['recall@k']:.4f}")
    print(f"F1@{K}        : {res['f1@k']:.4f}")
    print(f"MAP@{K}       : {res['map@k']:.4f}")
    print("=" * 60)

    # Simpan CSV
    summary_rows = [{
        "metode": "Cosine Similarity",
        "precision@k": res["precision@k"],
        "recall@k":    res["recall@k"],
        "f1@k":        res["f1@k"],
        "map@k":       res["map@k"],
    }]
    pd.DataFrame(summary_rows).to_csv(f"{RESULT_DIR}/evaluation_summary.csv", index=False)

    per_query_rows = []
    for qi, query in enumerate(queries):
        per_query_rows.append({
            "query": query,
            "precision": res["per_query"]["precision"][qi],
            "recall":    res["per_query"]["recall"][qi],
            "f1":        res["per_query"]["f1"][qi],
            "ap":        res["per_query"]["ap"][qi],
        })
    pd.DataFrame(per_query_rows).to_csv(f"{RESULT_DIR}/per_query_results.csv", index=False)
    print(f"\nHasil disimpan di folder {RESULT_DIR}/")

if __name__ == "__main__":
    main()