"""
Script utama evaluasi: bandingkan 4 metode similarity terhadap ground truth.

Jalankan: python evaluate.py
Output:
  - Tabel perbandingan di console
  - results/evaluation_summary.csv
  - results/per_query_results.csv
  - results/comparison_chart.png
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.preprocessing import preprocess, preprocess_to_tokens
from backend.similarity import compute_all_scores
from backend.evaluator import evaluate_method


# ============================================================
K = 10                  # top-K untuk evaluasi
ARTIFACT_DIR = "artifacts"
GT_PATH      = "ground_truth/labels.json"
RESULT_DIR   = "results"


# ============================================================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"EVALUASI 4 METODE SIMILARITY @K={K}")
    print("=" * 70)

    # Load artifacts
    print("\n[1] Loading artifacts...")
    with open(f"{ARTIFACT_DIR}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/processed_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    doc_ids        = docs["doc_ids"]
    titles         = docs["titles"]
    doc_tokens_all = docs["processed_tokens"]

    # Ground truth → {query: set(doc_ids)}
    ground_truth = {q: set(v["relevant_doc_ids"]) for q, v in gt_raw.items()}
    queries = list(ground_truth.keys())
    print(f"    Dokumen   : {len(doc_ids)}")
    print(f"    Query uji : {len(queries)}")

    # ============================================================
    # Untuk setiap query, hitung 4 skor → ranking per metode
    # ============================================================
    print(f"\n[2] Menghitung skor similarity untuk {len(queries)} query...")
    methods = ["cosine", "jaccard", "dice", "overlap"]
    rankings = {m: {} for m in methods}

    for qi, query in enumerate(queries, 1):
        q_tokens = preprocess_to_tokens(query)
        q_text   = " ".join(q_tokens)
        q_vec    = vectorizer.transform([q_text])

        scores = compute_all_scores(
            query_tokens=q_tokens,
            doc_tokens_list=doc_tokens_all,
            query_vec=q_vec,
            doc_matrix=tfidf_matrix,
        )

        for m in methods:
            if m not in scores:
                continue
            s = scores[m]
            # Ranking menurun
            ranked_idx = np.argsort(s)[::-1]
            ranked_ids = [int(doc_ids[i]) for i in ranked_idx]
            rankings[m][query] = ranked_ids

        print(f"    [{qi}/{len(queries)}] '{query[:55]}...' ✓")

    # ============================================================
    # Evaluasi tiap metode
    # ============================================================
    print(f"\n[3] Evaluasi setiap metode @K={K}...")
    results = {}
    for m in methods:
        results[m] = evaluate_method(rankings[m], ground_truth, k=K)

    # ============================================================
    # Tabel ringkasan
    # ============================================================
    print("\n" + "=" * 70)
    print("HASIL PERBANDINGAN (macro average)")
    print("=" * 70)
    header = f"{'Metode':<12} {'Precision@K':>14} {'Recall@K':>12} {'F1@K':>10} {'MAP@K':>10}"
    print(header)
    print("-" * 70)
    summary_rows = []
    for m in methods:
        r = results[m]
        row = {
            "metode": m,
            "precision@k": r["precision@k"],
            "recall@k":    r["recall@k"],
            "f1@k":        r["f1@k"],
            "map@k":       r["map@k"],
        }
        summary_rows.append(row)
        print(f"{m:<12} {r['precision@k']:>14.4f} {r['recall@k']:>12.4f} "
              f"{r['f1@k']:>10.4f} {r['map@k']:>10.4f}")

    # Cari yang terbaik
    best_m = max(results, key=lambda m: results[m]["f1@k"])
    print("-" * 70)
    print(f"🏆 Metode terbaik (F1@K): {best_m.upper()} "
          f"(F1={results[best_m]['f1@k']:.4f})")

    # ============================================================
    # Simpan CSV
    # ============================================================
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{RESULT_DIR}/evaluation_summary.csv", index=False)
    print(f"\n[4] Summary disimpan: {RESULT_DIR}/evaluation_summary.csv")

    # Per-query CSV
    per_query_rows = []
    for m in methods:
        per_q = results[m]["per_query"]
        for qi, query in enumerate(queries):
            per_query_rows.append({
                "metode": m,
                "query": query,
                "precision": per_q["precision"][qi],
                "recall":    per_q["recall"][qi],
                "f1":        per_q["f1"][qi],
                "ap":        per_q["ap"][qi],
            })
    df_per_query = pd.DataFrame(per_query_rows)
    df_per_query.to_csv(f"{RESULT_DIR}/per_query_results.csv", index=False)
    print(f"    Per-query disimpan: {RESULT_DIR}/per_query_results.csv")

    # ============================================================
    # Plot perbandingan
    # ============================================================
    print(f"\n[5] Membuat chart...")
    metrics = ["precision@k", "recall@k", "f1@k", "map@k"]
    labels  = ["Precision@K", "Recall@K", "F1@K", "MAP@K"]
    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(methods):
        vals = [results[m][k] for k in metrics]
        bars = ax.bar(x + i * width, vals, width, label=m.capitalize())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title(f"Perbandingan 4 Metode Similarity @K={K}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/comparison_chart.png", dpi=150)
    print(f"    Chart disimpan: {RESULT_DIR}/comparison_chart.png")

    print("\n" + "=" * 70)
    print("✅ EVALUASI SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()