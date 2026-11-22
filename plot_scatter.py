import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics.pairwise import cosine_similarity
from backend.preprocessing import preprocess_to_tokens

# ==========================================
# 1. MASUKKAN QUERY PANJANG DI SINI
# ==========================================
# Coba ganti dengan kalimat yang sangat panjang dan spesifik
QUERY_STRING = "Convolutional Neural Network CNN"

print(f"Memproses query: '{QUERY_STRING}'")

# 2. Load Data
with open("artifacts/vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
with open("artifacts/tfidf_matrix.pkl", "rb") as f: tfidf_matrix = pickle.load(f)
with open("artifacts/processed_docs.pkl", "rb") as f: docs = pickle.load(f)
with open("ground_truth/labels.json", "r", encoding="utf-8") as f: gt_raw = json.load(f)

doc_ids = docs["doc_ids"]

# Kita tidak pakai ground truth untuk query manual, tapi kita tetap butuh 
# untuk mewarnai titik (jika kebetulan dokumen itu ada di ground truth query lain)
# Sebagai gantinya, kita akan mewarnai berdasarkan skor threshold (misal > 0.3 dianggap relevan)

# 3. Hitung Cosine Similarity
q_tokens = preprocess_to_tokens(QUERY_STRING)
q_text = " ".join(q_tokens)
q_vec = vectorizer.transform([q_text])
cos_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

# 4. Ambil Top 20 Dokumen
top_n = 20
top_indices = np.argsort(cos_scores)[::-1][:top_n]

ranks = np.arange(1, top_n + 1)
scores = cos_scores[top_indices]

# Beri warna biru jika skor > 0.3 (anggap relevan), merah jika di bawahnya
# Kamu bisa mengubah batas 0.3 ini sesuai kebutuhan
colors = ['#0072BD' if score > 0.25 else 'red' for score in scores]

# 5. Plotting
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(ranks, scores, color='gray', linestyle='--', alpha=0.5, zorder=0)
plt.scatter(ranks, scores, c=colors, s=80, alpha=0.9, edgecolors='black', linewidth=0.5, zorder=1)

plt.xticks(ranks)
plt.xlabel('Ranking (Top 20)', fontsize=12, fontweight='bold')
plt.ylabel('Cosine Similarity Score', fontsize=12, fontweight='bold')

# Memotong query agar judul grafik tidak kepanjangan
title_query = QUERY_STRING[:50] + "..." if len(QUERY_STRING) > 50 else QUERY_STRING
plt.title(f'Hasil Pencarian Top-20\nQuery: "{title_query}"', fontsize=13, pad=15)

# Set batas Y maksimal 1.0 agar proporsional
plt.ylim(0, 1.0) 

plt.tick_params(direction='in', top=True, right=True, length=5, width=1)

blue_dot = mlines.Line2D([], [], color='#0072BD', marker='o', linestyle='None', markersize=8, label='Skor Tinggi (Relevan)')
red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Skor Rendah (Tidak Relevan)')
plt.legend(handles=[blue_dot, red_dot], loc='upper right', frameon=False, fontsize=11)

plt.tight_layout()
plt.savefig('rank_vs_score_panjang_Pendek.png')
plt.show()