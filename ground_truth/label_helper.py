"""
GROUND TRUTH LABEL HELPER v2.1
==============================
Fix: handle file labels.json kosong/corrupt.

Kriteria relevansi:
  ✅ RELEVAN   : Dokumen membahas topik/metode utama yang sama dengan query
  ❌ TIDAK     : Sekadar ada kata yang sama, tapi topik berbeda
  ⚠️  RAGU     : Boleh tandai n (tidak relevan)

Command:
  y = relevan
  n = tidak relevan (default, tekan Enter)
  s = skip query ini
  q = save & keluar
"""
import json
import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.preprocessing import preprocess
from backend.similarity import cosine_scores


# ============================================================
# QUERY UJI
# ============================================================
QUERIES = [
    "klasifikasi citra tanaman menggunakan CNN deep learning",
    "analisis sentimen media sosial Indonesia",
    "sistem pakar diagnosa penyakit menggunakan forward chaining",
    "sistem pendukung keputusan dengan metode AHP",
    "perancangan UI/UX aplikasi mobile",
    "prediksi time series dengan LSTM",
    "deteksi objek menggunakan YOLO",
    "optimasi proses bisnis lean six sigma",
    "sistem IoT monitoring berbasis sensor",
    "analisis pengenalan wajah face recognition",
]

TOP_K_CANDIDATES = 10
LABELS_PATH      = "ground_truth/labels.json"
ABSTRACT_PREVIEW = 250


# ============================================================
# IO — DEFENSIVE
# ============================================================
def load_existing_labels() -> dict:
    """Load labels jika ada; kalau kosong/corrupt return {}."""
    if not os.path.exists(LABELS_PATH):
        return {}

    if os.path.getsize(LABELS_PATH) == 0:
        print(f"⚠️  {LABELS_PATH} kosong (0 byte), dianggap belum ada label.")
        return {}

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"⚠️  {LABELS_PATH} hanya whitespace, dianggap belum ada label.")
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"⚠️  {LABELS_PATH} corrupt ({e}), dianggap belum ada label.")
        return {}


def save_labels(labels: dict):
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


def preview(text: str, n: int = ABSTRACT_PREVIEW) -> str:
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("GROUND TRUTH LABEL HELPER v2.1")
    print("=" * 70)
    print("""
  KRITERIA RELEVANSI:
    ✅ RELEVAN    : Dokumen membahas topik/metode UTAMA yang sama dengan query
    ❌ TIDAK      : Hanya ada kemiripan kata, topik berbeda
    🟡 RAGU       : Tandai n (tidak relevan) supaya konservatif

  INPUT:
    y  → relevan
    n  → tidak relevan (atau langsung Enter)
    s  → skip query ini
    q  → save & keluar
""")
    input("Tekan Enter untuk mulai...")

    # Load artifacts
    print("\nLoading artifacts...")
    with open("artifacts/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("artifacts/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("artifacts/processed_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    import pandas as pd
    df_full = pd.read_csv("artifacts/processed_dataset.csv")
    abstracts = df_full["abstrak"].fillna("").tolist()

    print(f"Total dokumen: {len(docs['doc_ids'])}")

    labels = load_existing_labels()
    if labels:
        print(f"\n⚠️  Sudah ada {len(labels)} query di {LABELS_PATH}")
        print("   Query yang sudah dilabel akan di-skip.")

    for qi, query in enumerate(QUERIES):
        if query in labels:
            n_rel = labels[query]["num_relevant"]
            print(f"\n[SKIP] Query {qi+1} sudah dilabel ({n_rel} relevan): {query[:60]}...")
            continue

        print("\n" + "=" * 70)
        print(f"QUERY {qi+1}/{len(QUERIES)}: {query}")
        print("=" * 70)

        q_proc = preprocess(query)
        q_vec = vectorizer.transform([q_proc])
        scores = cosine_scores(q_vec, tfidf_matrix)
        ranked_idx = scores.argsort()[::-1][:TOP_K_CANDIDATES]

        relevant_ids = []
        skip_query = False

        for rank, idx in enumerate(ranked_idx, 1):
            doc_id  = int(docs["doc_ids"][idx])
            judul   = docs["titles"][idx]
            tahun   = docs["tahun"][idx]
            penulis = docs["penulis"][idx]
            skor    = float(scores[idx])
            abstrak = preview(abstracts[idx])

            print(f"\n[{rank}/{TOP_K_CANDIDATES}] doc_id={doc_id} | {tahun} | skor_cos={skor:.4f}")
            print(f"  Judul    : {judul}")
            print(f"  Penulis  : {penulis}")
            print(f"  Abstrak  : {abstrak}")

            ans = input("  → Relevan? [y/n/s/q] > ").strip().lower()

            if ans == "q":
                save_labels(labels)
                print(f"\n✅ Saved {len(labels)} query ke {LABELS_PATH}. Keluar.")
                return
            elif ans == "s":
                skip_query = True
                break
            elif ans == "y":
                relevant_ids.append(doc_id)

        if skip_query:
            print(f"\n[SKIP] Query '{query[:50]}...' di-skip")
            continue

        labels[query] = {
            "relevant_doc_ids": relevant_ids,
            "num_relevant": len(relevant_ids),
        }
        save_labels(labels)
        print(f"\n✅ Query '{query[:50]}...' → {len(relevant_ids)} dokumen relevan (autosaved)")

    print("\n" + "=" * 70)
    print(f"✅ SELESAI. Ground truth: {LABELS_PATH}")
    print("=" * 70)
    total_rel = sum(v["num_relevant"] for v in labels.values())
    print(f"Total query              : {len(labels)}")
    print(f"Total dokumen relevan    : {total_rel}")
    print(f"Rata-rata relevan/query  : {total_rel / max(len(labels),1):.1f}")


if __name__ == "__main__":
    main()