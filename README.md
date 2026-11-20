# Sistem Temu Kembali Informasi Skripsi Fasilkom UNEJ

Sistem pencarian dan deteksi kemiripan judul/abstrak skripsi Fasilkom Universitas Jember menggunakan TF-IDF dan perbandingan empat metode similarity: **Cosine Similarity**, **Jaccard Coefficient**, **Sørensen-Dice**, dan **Overlap Similarity**.

Project ini merupakan implementasi UTS mata kuliah Information Retrieval.

---

## Fitur

- Preprocessing teks Bahasa Indonesia (case folding, cleaning, tokenisasi, stopword removal, stemming Sastrawi)
- Pembobotan TF-IDF dengan unigram + bigram
- Pencarian dokumen berbasis query bebas
- Perbandingan 4 metode similarity secara side-by-side
- Evaluasi performa dengan Precision@K, Recall@K, F1@K, dan MAP@K
- Antarmuka web interaktif (FastAPI + Tailwind CSS)
- Visualisasi hasil evaluasi (tabel + bar chart)

---

## Teknologi

| Komponen | Teknologi |
|---|---|
| Bahasa | Python 3.10+ |
| Backend | FastAPI, Uvicorn |
| NLP | Sastrawi, NLTK, scikit-learn |
| Frontend | HTML, Tailwind CSS (CDN), Chart.js (CDN) |
| Data | CSV |

## Instalasi

### 1. Clone / siapkan project

Pastikan folder `data/dataset_skripsi_fasilkom_clean.csv` sudah ada.

### 2. Install dependensi

```bash
pip install -r requirements.txt
Isi requirements.txt:

text
fastapi
uvicorn[standard]
pandas
numpy
scikit-learn
Sastrawi
nltk
python-multipart
jinja2
matplotlib
3. Download stopwords NLTK
bash
python -c "import nltk; nltk.download('stopwords')"
Cara Menjalankan
Tahap 1: Build Index TF-IDF
Preprocessing 587 dokumen dengan stemming Sastrawi (estimasi 25-30 menit pertama kali). Hasil disimpan ke artifacts/ sebagai cache.

bash
python build_index.py
Output:

artifacts/vectorizer.pkl

artifacts/tfidf_matrix.pkl (587 x 12.601 fitur)

artifacts/processed_docs.pkl

artifacts/processed_texts.pkl (checkpoint preprocessing)

Tahap 2: Labeling Ground Truth
Script interaktif untuk melabeli 8-10 query uji. Untuk setiap query, sistem menampilkan 10 kandidat dokumen teratas (via Cosine Similarity) untuk dilabeli relevan atau tidak.

bash
python ground_truth/label_helper.py
Perintah input:

y - dokumen relevan

n atau Enter - tidak relevan

s - skip query

q - save dan keluar

Progress auto-saved setiap selesai satu query.

Tahap 3: Evaluasi
Membandingkan performa 4 metode similarity terhadap ground truth.

bash
python evaluate.py
Output:

results/evaluation_summary.csv

results/per_query_results.csv

results/comparison_chart.png

Tahap 4: Jalankan Server Web
bash
python -m uvicorn backend.main:app --reload --port 8000
Buka browser: http://localhost:8000

Endpoint API
GET /api/health
Cek status server dan jumlah dokumen yang di-load.

json
{ "status": "ok", "docs": 587 }
POST /api/search
Pencarian dokumen dengan query bebas.

Request:

json
{
  "query": "klasifikasi citra tanaman CNN",
  "top_k": 10
}
Response:

json
{
  "query": "klasifikasi citra tanaman CNN",
  "processed_query": "klasifikasi citra tanaman cnn",
  "top_k": 10,
  "elapsed_ms": 245.3,
  "results": [
    {
      "rank": 1,
      "doc_id": 60,
      "judul": "...",
      "tahun": 2024,
      "penulis": "...",
      "kata_kunci": "...",
      "abstrak": "...",
      "scores": {
        "cosine": 0.7421,
        "jaccard": 0.3125,
        "dice": 0.4762,
        "overlap": 0.5294
      }
    }
  ]
}
GET /api/evaluation
Ambil hasil evaluasi (summary + per-query + ground truth).
