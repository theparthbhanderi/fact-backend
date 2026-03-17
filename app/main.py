"""
AI Fact-Checker — FastAPI Application Entry Point.

This module initializes the FastAPI application, configures CORS middleware,
logging, and registers all API route modules.

Pipeline Overview:
    User Claim → /api/fact-check → News Search → Article Extraction
    → Embedding Generation → Vector Storage → RAG Retrieval → LLM Analysis
    → Verdict + Explanation + Evidence → Frontend Dashboard
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

from app.database import engine, Base
from app.routes.factcheck import router as factcheck_router
from app.routes.ocr_factcheck import router as ocr_factcheck_router
from app.routes.history import router as history_router
from app.routes.analytics import router as analytics_router
from app.routes.url_factcheck import router as url_factcheck_router
from app.routes.translation import router as translation_router
from app.routes.news_search import router as news_router

# ── Logging Configuration ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Initialize FastAPI App ─────────────────────────────────────────────
app = FastAPI(
    title="AI Fact-Checker API",
    description=(
        "An AI-powered fact-checking system that searches the internet, "
        "retrieves evidence articles, analyzes them using an LLM and RAG "
        "pipeline, and returns a verdict with explanation and sources."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ────────────────────────────────────────────────────
# Allow the frontend dev server (and any other configured origins) to
# communicate with the backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Database ───────────────────────────────────────────────
logger.info("🗄️ Initializing SQLite database...")
Base.metadata.create_all(bind=engine)

# ── Register Routes ───────────────────────────────────────────────────
app.include_router(factcheck_router, prefix="/api")
app.include_router(ocr_factcheck_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(url_factcheck_router, prefix="/api")
app.include_router(translation_router, prefix="/api")
app.include_router(news_router, prefix="/api")


# ── Health Check ──────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """Health-check endpoint. Returns API status."""
    return {
        "status": "online",
        "service": "AI Fact-Checker API",
        "version": "0.1.0",
    }
