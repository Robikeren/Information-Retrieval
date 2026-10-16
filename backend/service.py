"""
Business logic: load artifacts, jalankan search untuk 4 metode sekaligus.
"""
import json
import math
import os
import pickle
import time

import numpy as np

from backend.preprocessing import preprocess, preprocess_to_tokens
from backend.similarity import compute_all_scores


# ============================================================
# PATH
# ============================================================
ARTIFACT_DIR = "artifacts"
GT_PATH      = "ground_truth/labels.json"
RESULTS_DIR  = "results"


# ============================================================
# HELPERS — handle NaN/None
# ============================================================
def safe_str(v) -> str:
    """Convert NaN/None → string kosong, else str(v)."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v)


def safe_int(v):
    """Convert ke int, NaN/None → None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def safe_float(v, digits: int = 4) -> float:
    """Convert ke float rounded; NaN → 0.0."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return round(f, digits)
    except (ValueError, TypeError):
        return 0.0


# ============================================================
# SINGLETON STATE (load sekali saat startup)
# ============================================================
class SearchEngine:
    def __init__(self):
        self._loaded = False
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = None
        self.titles = None
        self.tahun = None
        self.penulis = None
        self.kata_kunci = None
        self.abstracts = None
        self.doc_tokens = None

    def load(self):
        if self._loaded:
            return
        t0 = time.time()
        print("Loading artifacts...")

        with open(f"{ARTIFACT_DIR}/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(f"{ARTIFACT_DIR}/tfidf_matrix.pkl", "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(f"{ARTIFACT_DIR}/processed_docs.pkl", "rb") as f:
            d = pickle.load(f)
        self.doc_ids      = d["doc_ids"]
        self.titles       = d["titles"]
        self.tahun        = d["tahun"]
        self.penulis      = d["penulis"]
        self.kata_kunci   = d["kata_kunci"]
        self.doc_tokens   = d["processed_tokens"]

        import pandas as pd
        df = pd.read_csv(f"{ARTIFACT_DIR}/processed_dataset.csv")
        self.abstracts = df["abstrak"].fillna("").tolist()

        self._loaded = True
        print(f"Loaded in {time.time() - t0:.2f}s | {len(self.doc_ids)} docs")

    # --------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> dict:
        self.load()

        t0 = time.time()
        q_tokens = preprocess_to_tokens(query)
        q_text   = " ".join(q_tokens)
        q_vec    = self.vectorizer.transform([q_text])

        scores = compute_all_scores(
            query_tokens=q_tokens,
            doc_tokens_list=self.doc_tokens,
            query_vec=q_vec,
            doc_matrix=self.tfidf_matrix,
        )

        ranked_idx = np.argsort(scores["cosine"])[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_idx, 1):
            abstrak_full = safe_str(self.abstracts[idx]) if idx < len(self.abstracts) else ""
            results.append({
                "rank": rank,
                "doc_id": safe_int(self.doc_ids[idx]),
                "judul": safe_str(self.titles[idx]),
                "tahun": safe_int(self.tahun[idx]),
                "penulis": safe_str(self.penulis[idx]),
                "kata_kunci": safe_str(self.kata_kunci[idx]),
                "abstrak": abstrak_full[:400] + ("..." if len(abstrak_full) > 400 else ""),
                "scores": {
                    "cosine":  safe_float(scores["cosine"][idx]),
                    "jaccard": safe_float(scores["jaccard"][idx]),
                    "dice":    safe_float(scores["dice"][idx]),
                    "overlap": safe_float(scores["overlap"][idx]),
                },
            })

        elapsed = (time.time() - t0) * 1000
        return {
            "query": query,
            "processed_query": q_text,
            "top_k": top_k,
            "elapsed_ms": round(elapsed, 1),
            "results": results,
        }

    # --------------------------------------------------------
    def load_evaluation(self) -> dict:
        summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
        per_query_path = os.path.join(RESULTS_DIR, "per_query_results.csv")
        gt_path = GT_PATH

        if not os.path.exists(summary_path):
            return {"available": False, "message": "Belum ada hasil evaluasi. Jalankan evaluate.py dulu."}

        import pandas as pd
        summary = pd.read_csv(summary_path).fillna(0).to_dict(orient="records")
        per_query = pd.read_csv(per_query_path).fillna(0).to_dict(orient="records")
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        return {
            "available": True,
            "k": 10,
            "summary": summary,
            "per_query": per_query,
            "ground_truth": gt,
        }


# Singleton
engine = SearchEngine()