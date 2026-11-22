import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data evaluasi
df = pd.read_csv('results/evaluation_summary.csv')
row = df.iloc[0] # Ambil baris pertama (Cosine Similarity)

metrics = ['Precision@K', 'Recall@K', 'F1@K', 'MAP@K']
values = [row['precision@k'], row['recall@k'], row['f1@k'], row['map@k']]
colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E'] # Warna standar MATLAB/scientific

plt.figure(figsize=(8, 5), dpi=300)
bars = plt.bar(metrics, values, color=colors, width=0.6, edgecolor='black', linewidth=0.7)

# Tambahkan teks nilai di atas bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.ylim(0, 1.1)
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Evaluasi Kinerja Sistem (Cosine Similarity @K=10)', fontsize=14, fontweight='bold', pad=15)
plt.tick_params(direction='in', top=True, right=True, length=5, width=1)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('evaluation_bar_chart.png')
plt.show()