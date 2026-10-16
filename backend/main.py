"""
FastAPI app: backend untuk sistem pencarian & perbandingan similarity.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.service import engine


# ============================================================
app = FastAPI(title="IR Skripsi Fasilkom UNEJ", version="1.0")

# CORS (biar frontend lokal bisa akses)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST/RESPONSE SCHEMA
# ============================================================
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Kata kunci pencarian")
    top_k: int = Field(10, ge=1, le=50, description="Jumlah hasil")


# ============================================================
# ROUTES API
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


@app.get("/api/evaluation")
def evaluation():
    return engine.load_evaluation()


# ============================================================
# SERVE FRONTEND
# ============================================================
FRONTEND_DIR = "frontend"


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
def startup():
    engine.load()