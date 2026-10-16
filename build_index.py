"""
BUILD INDEX v2 — robust + optimized
- Picklable TfidfVectorizer (pakai token_pattern regex, bukan lambda)
- Batch stemming: stem hanya token unik (3-5x lebih cepat)
- Checkpoint: simpan hasil preprocessing sebelum TF-IDF
"""
import os
import pickle
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.preprocessing import (
    case_folding, clean_text, tokenize, remove_stopwords, stem_word,
)


INPUT_CSV        = "data/dataset_skripsi_fasilkom_clean.csv"
ARTIFACT_DIR     = "artifacts"
PREPROCESS_CACHE = f"{ARTIFACT_DIR}/processed_texts.pkl"


# ============================================================
# BATCH PREPROCESSING (optimized)
# ============================================================
def preprocess_corpus_batch(texts: list[str]) -> list[str]:
    print("  [Phase 1/3] Case folding + cleaning + tokenizing + stopword...")
    t0 = time.time()
    tokenized = []
    unique_tokens = set()
    for i, text in enumerate(texts):
        t = case_folding(text or "")
        t = clean_text(t)
        toks = remove_stopwords(tokenize(t))
        tokenized.append(toks)
        unique_tokens.update(toks)
        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{len(texts)} ({time.time()-t0:.1f}s)")

    print(f"  [Phase 2/3] Stemming {len(unique_tokens)} unique tokens (Sastrawi)...")
    t0 = time.time()
    stem_map = {}
    for i, tok in enumerate(unique_tokens):
        stem_map[tok] = stem_word(tok)
        if (i + 1) % 2000 == 0:
            print(f"    ... {i+1}/{len(unique_tokens)} ({time.time()-t0:.1f}s)")

    print(f"  [Phase 3/3] Apply stem map ke semua dokumen...")
    processed = []
    for toks in tokenized:
        stemmed = [stem_map.get(t, t) for t in toks]
        stemmed = [s for s in stemmed if s]
        processed.append(" ".join(stemmed))
    return processed


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("=" * 60)
    print("BUILD INDEX v2 — TF-IDF SKRIPSI FASILKOM UNEJ")
    print("=" * 60)

    # 1. Load
    df = pd.read_csv(INPUT_CSV)
    print(f"\n[1] Load dataset: {len(df)} baris")

    # 2. Drop tanpa abstrak
    df = df[df["abstrak_missing"] == False].reset_index(drop=True)
    print(f"[2] Drop tanpa abstrak: {len(df)} baris")

    # 3. Preprocessing (dengan checkpoint)
    if os.path.exists(PREPROCESS_CACHE):
        print(f"\n[3] Load preprocessed dari checkpoint: {PREPROCESS_CACHE}")
        with open(PREPROCESS_CACHE, "rb") as f:
            processed = pickle.load(f)
        print(f"    Loaded {len(processed)} dokumen (skip preprocessing)")
    else:
        print(f"\n[3] Preprocessing {len(df)} dokumen...")
        t0 = time.time()
        processed = preprocess_corpus_batch(df["text_for_retrieval"].fillna("").tolist())
        print(f"    Selesai dalam {time.time()-t0:.1f}s")
        with open(PREPROCESS_CACHE, "wb") as f:
            pickle.dump(processed, f)
        print(f"    ✅ Checkpoint disimpan: {PREPROCESS_CACHE}")

    df["processed"] = processed
    df["processed_tokens"] = df["processed"].apply(lambda t: t.split() if t else [])

    # 4. Drop yang kosong
    before = len(df)
    df = df[df["processed"].str.len() > 0].reset_index(drop=True)
    print(f"\n[4] Drop kosong: {before} → {len(df)}")

    # 5. TF-IDF — VERSI PICKLABLE (no lambda!)
    print(f"\n[5] Build TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        token_pattern=r"\S+",     # split on whitespace (PICKLABLE, string bukan lambda)
        lowercase=False,          # sudah lowercase dari preprocessing
        preprocessor=None,
        tokenizer=None,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(df["processed"])
    print(f"    Matrix shape: {tfidf_matrix.shape}")
    print(f"    Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 6. Save artifacts
    print(f"\n[6] Simpan artifacts ke {ARTIFACT_DIR}/")

    # Buang file corrupt dari run sebelumnya
    for fname in ["vectorizer.pkl", "tfidf_matrix.pkl", "processed_docs.pkl"]:
        fpath = f"{ARTIFACT_DIR}/{fname}"
        if os.path.exists(fpath):
            os.remove(fpath)

    with open(f"{ARTIFACT_DIR}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{ARTIFACT_DIR}/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(f"{ARTIFACT_DIR}/processed_docs.pkl", "wb") as f:
        pickle.dump({
            "doc_ids":          df["doc_id"].tolist(),
            "titles":           df["judul"].tolist(),
            "tahun":            df["tahun"].tolist(),
            "penulis":          df["penulis"].tolist(),
            "kata_kunci":       df["kata_kunci"].tolist(),
            "processed":        df["processed"].tolist(),
            "processed_tokens": df["processed_tokens"].tolist(),
        }, f)
    df.to_csv(f"{ARTIFACT_DIR}/processed_dataset.csv", index=False)

    print(f"\n✅ Selesai. Total dokumen terindeks: {len(df)}")


if __name__ == "__main__":
    main()