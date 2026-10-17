"""
FastAPI app: backend untuk sistem pencarian & perbandingan similarity.
"""
import os
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.service import engine, safe_str, safe_int


# ============================================================
app = FastAPI(title="IR Skripsi Fasilkom UNEJ", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SCHEMA
# ============================================================
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(10, ge=1, le=50)


# ============================================================
# API ROUTES
# ============================================================
@app.get("/api/health")
def health():
    return {"status": "ok", "docs": len(engine.doc_ids) if engine._loaded else 0}


@app.post("/api/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query kosong")
    try:
        return engine.search(req.query, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/random")
def random_docs(n: int = 10):
    """Ambil n dokumen acak untuk ditampilkan saat halaman pertama dibuka."""
    if not engine._loaded:
        engine.load()
    total = len(engine.doc_ids)
    n = max(1, min(n, total))
    indices = random.sample(range(total), n)

    results = []
    for i, idx in enumerate(indices, 1):
        abstrak = safe_str(engine.abstracts[idx]) if idx < len(engine.abstracts) else ""
        results.append({
            "rank": i,
            "doc_id": safe_int(engine.doc_ids[idx]),
            "judul": safe_str(engine.titles[idx]),
            "tahun": safe_int(engine.tahun[idx]),
            "penulis": safe_str(engine.penulis[idx]),
            "kata_kunci": safe_str(engine.kata_kunci[idx]),
            "abstrak": abstrak[:400] + ("..." if len(abstrak) > 400 else ""),
            "scores": {"cosine": 0.0, "jaccard": 0.0, "dice": 0.0, "overlap": 0.0},
        })
    return {"n": n, "results": results}


@app.get("/api/evaluation")
def evaluation():
    return engine.load_evaluation()


# ============================================================
# STATIC / ASSETS
# ============================================================
FRONTEND_DIR = "frontend"


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/logo.png")
def logo():
    """Serve logo Unej dari root project."""
    path = "logo.png"
    if not os.path.exists(path):
        return Response(status_code=204)
    return FileResponse(path, media_type="image/png")

@app.get("/bg.png")
def background():
    """Serve gambar background dari root project."""
    path = "bg.png"
    if not os.path.exists(path):
        return Response(status_code=204)
    return FileResponse(path, media_type="image/png")


@app.get("/favicon.ico")
def favicon():
    """Serve favicon dari logo.png (biar log bersih)."""
    path = "logo.png"
    if not os.path.exists(path):
        return Response(status_code=204)
    return FileResponse(path, media_type="image/png")


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============================================================
@app.on_event("startup")
def startup():
    engine.load()