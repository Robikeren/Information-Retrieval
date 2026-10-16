"""
Pipeline preprocessing teks Bahasa Indonesia untuk dataset skripsi Fasilkom UNEJ.

Tahapan:
1. Case folding (lowercase)
2. Cleaning (hapus URL, angka, tanda baca, whitespace ganda)
3. Tokenisasi
4. Stopword removal (NLTK Indonesian + custom akademik)
5. Stemming (Sastrawi)
"""
import re
import string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords


# ============================================================
# INISIALISASI (sekali saja)
# ============================================================
_stemmer = StemmerFactory().create_stemmer()

# Cache global stem (lebih cepat dari lru_cache untuk string pendek)
_STEM_CACHE: dict[str, str] = {}

# Stopword NLTK Indonesia + tambahan konteks akademik
_BASE_STOPWORDS = set(stopwords.words("indonesian"))

# Kata-kata umum di judul skripsi yang tidak diskriminatif topik
_ACADEMIC_STOPWORDS = {
    "sistem", "aplikasi", "penelitian", "analisis", "berbasis",
    "menggunakan", "metode", "studi", "kasus", "implementasi",
    "rancang", "bangun", "pengembangan", "perancangan", "desain",
    "evaluasi", "penerapan", "pada", "untuk", "dengan", "dan",
    "di", "ke", "dari", "yang", "ini", "itu", "adalah", "atau",
    "sebagai", "dalam", "akan", "telah", "dapat", "oleh", "secara",
    "hasil", "dilakukan", "terhadap", "berbagai", "serta", "juga",
    "agar", "karena", "namun", "sehingga", "apabila", "bila",
    "suatu", "para", "setiap", "seluruh", "semua", "antara",
}

STOPWORDS = _BASE_STOPWORDS | _ACADEMIC_STOPWORDS


# ============================================================
# FUNGSI UTAMA
# ============================================================
def case_folding(text: str) -> str:
    return text.lower()


def clean_text(text: str) -> str:
    """Hapus URL, email, angka, tanda baca, whitespace ganda."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)                    # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^\x00-\x7F]+", " ", text)            # non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def stem_word(word: str) -> str:
    """Cache global stem (lebih cepat dari lru_cache untuk string pendek)."""
    cached = _STEM_CACHE.get(word)
    if cached is not None:
        return cached
    result = _stemmer.stem(word)
    _STEM_CACHE[word] = result
    return result


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem_word(t) for t in tokens]


def preprocess(text: str, do_stemming: bool = True) -> str:
    """Pipeline lengkap → string token terpisah spasi."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = case_folding(text)
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if do_stemming:
        tokens = stem_tokens(tokens)
    tokens = [t for t in tokens if t]  # buang hasil stem kosong
    return " ".join(tokens)


def preprocess_to_tokens(text: str, do_stemming: bool = True) -> list[str]:
    """Sama seperti preprocess() tapi return list token (untuk Jaccard/Dice/Overlap)."""
    return preprocess(text, do_stemming).split()