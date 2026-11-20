"""
Business logic: load artifacts, jalankan search menggunakan Cosine Similarity.
"""
import math
import os
import pickle
import time
import numpy as np

from backend.preprocessing import preprocess_to_tokens
from backend.similarity import cosine_scores

ARTIFACT_DIR = "artifacts"

def safe_str(v) -> str:
    if v is None: return ""
    if isinstance(v, float) and math.isnan(v): return ""
    return str(v)

def safe_int(v):
    if v is None: return None
    if isinstance(v, float) and math.isnan(v): return None
    try: return int(v)
    except (ValueError, TypeError): return None

def safe_float(v, digits: int = 4) -> float:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f): return 0.0
        return round(f, digits)
    except (ValueError, TypeError): return 0.0

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

    def load(self):
        if self._loaded: return
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

        import pandas as pd
        df = pd.read_csv(f"{ARTIFACT_DIR}/processed_dataset.csv")
        self.abstracts = df["abstrak"].fillna("").tolist()

        self._loaded = True
        print(f"Loaded in {time.time() - t0:.2f}s | {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> dict:
        self.load()
        t0 = time.time()
        
        # 1. Preprocessing Query
        q_tokens = preprocess_to_tokens(query)
        q_text   = " ".join(q_tokens)
        q_vec    = self.vectorizer.transform([q_text])

        # 2. Hitung Cosine Similarity
        scores = cosine_scores(q_vec, self.tfidf_matrix)

        # 3. Ranking
        ranked_idx = np.argsort(scores)[::-1][:top_k]

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
                "score": safe_float(scores[idx]), # Hanya 1 skor
            })

        elapsed = (time.time() - t0) * 1000
        return {
            "query": query,
            "processed_query": q_text,
            "top_k": top_k,
            "elapsed_ms": round(elapsed, 1),
            "results": results,
        }

# Singleton
engine = SearchEngine()