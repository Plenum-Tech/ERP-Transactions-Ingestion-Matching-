# app/main.py
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from app.db import init_pool, close_pool
from app.routers import tx_ingest
from app.routers import match_preview  # keep your name
from app.routers import lines_explain
from app.routers import tx_bank_reco  # <- this file






from app.utils.logger import RequestContextLogMiddleware, get_logger
import os

app = FastAPI(title="Transactions Ingestion & Matching (POC)")

# logging middleware
app.add_middleware(RequestContextLogMiddleware)

# Static hosting for uploaded files (used by /tx/upload)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.on_event("startup")
async def _startup():
    await init_pool()
    get_logger(__name__).info("startup.complete")

@app.on_event("shutdown")
async def _shutdown():
    await close_pool()
    get_logger(__name__).info("shutdown.complete")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# Ingestion routes (bank statements + receipts + upload)
app.include_router(tx_ingest.router)
app.include_router(match_preview.router)
app.include_router(lines_explain.router)
app.include_router(tx_bank_reco.router)