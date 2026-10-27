from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Configuration container for the RAG Astronomy Knowledge Assistant.

    All retrieval parameters are surfaced as class-level attributes so the
    module-level backwards-compatible exports below stay in sync automatically.
    """

    # ==================== API KEYS ====================
    GE_API_KEY: str | None = os.getenv("GE_API_KEY") or os.getenv("LLM_API_KEY")
    LLM_API_KEY: str | None = GE_API_KEY

    # ==================== ENDPOINTS ====================
    _raw_base = os.getenv("LLM_BASE_URL")
    LLM_BASE_URL: str = _raw_base.strip() if _raw_base else ""
    OPENAI_BASE_URL: str = LLM_BASE_URL

    # ==================== MODELS ====================
    _raw_llm_model = os.getenv("LLM_MODEL")
    _raw_embed_model = os.getenv("EMBEDDING_MODEL")

    EMBEDDING_MODEL: str = _raw_embed_model.strip() if _raw_embed_model else ""
    LLM_MODEL: str = _raw_llm_model.strip() if _raw_llm_model else ""
    _raw_image_model = os.getenv("IMAGE_ANALYSIS_MODEL")
    IMAGE_ANALYSIS_MODEL: str = _raw_image_model.strip() if _raw_image_model else ""
    ENABLE_IMAGE_ANALYSIS: bool = os.getenv("ENABLE_IMAGE_ANALYSIS", "false").lower() in ("1", "true", "yes")

    # ==================== POSTGRESQL ====================
    USE_POSTGRES: bool = os.getenv("USE_POSTGRES", "false").lower() in ("1", "true", "yes")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_helpdesk")
    DATABASE_ECHO: bool = os.getenv("DATABASE_ECHO", "false").lower() in ("1", "true", "yes")

    # ==================== PATHS ====================
    BASE_DIR: Path = BASE_DIR
    KNOWLEDGE_BASE_DIR: Path = BASE_DIR / "knowledge_base"
    VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"

    COLLECTION_NAME: str = "knowledge_base"

    # ==================== CHUNKING ====================
    # Canonical chunking is token-based (see document_loader TARGET_TOKENS).
    # CHUNK_SIZE / CHUNK_OVERLAP are legacy pipeline field names: target and max tokens.
    CHUNK_SIZE: int = int(
        os.getenv("CHUNK_TARGET_TOKENS", os.getenv("CHUNK_SIZE", "350"))
    )
    CHUNK_MAX_TOKENS: int = int(
        os.getenv("CHUNK_MAX_TOKENS", "450")
    )
    CHUNK_OVERLAP: int = int(
        os.getenv(
            "OVERSIZED_PARAGRAPH_OVERLAP",
            os.getenv("CHUNK_OVERLAP", "40"),
        )
    )
    MIN_CHUNK_LENGTH: int = 50

    # ==================== EMBEDDING INGESTION ====================
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "2"))
    EMBEDDING_RETRY_BASE_DELAY: float = float(
        os.getenv("EMBEDDING_RETRY_BASE_DELAY", "1.0")
    )

    # ==================== RETRIEVAL ====================
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "10"))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "10"))
    HYBRID_TOP_K: int = int(os.getenv("HYBRID_TOP_K", "12"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))
    TOP_K: int = RERANK_TOP_K

    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.35"))
    MIN_RERANK_SCORE: float = 0.5

    # ==================== HYBRID WEIGHTS ====================
    BM25_WEIGHT: float = 0.3
    VECTOR_WEIGHT: float = 0.7
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # ==================== RERANKER ====================
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"
    )
    ENABLE_RERANKING: bool = True

    # ==================== ASTRONOMY DOMAIN KEYWORDS ====================
    # Used for light query categorisation and document classification.

    PLANETARY_KEYWORDS = [
        "planet", "planets", "orbit", "atmosphere", "moon", "moons",
        "asteroid", "asteroids", "comet", "comets", "crater", "craters",
        "solar system", "mercury", "venus", "earth", "mars", "jupiter",
        "saturn", "uranus", "neptune", "dwarf planet", "pluto",
    ]

    ASTROPHYSICS_KEYWORDS = [
        "star", "stars", "sun", "solar", "black hole", "blackhole",
        "singularity", "event horizon", "supernova", "neutron star",
        "pulsar", "white dwarf", "red giant", "stellar", "gravity",
        "luminosity", "spectral", "main sequence", "fusion",
    ]

    COSMOLOGY_KEYWORDS = [
        "galaxy", "galaxies", "milky way", "universe", "expansion",
        "big bang", "dark matter", "dark energy", "cosmic", "cosmology",
        "redshift", "quasar", "hubble constant", "inflation",
        "large scale structure",
    ]

    MISSIONS_KEYWORDS = [
        "isro", "chandrayaan", "pragyan", "vikram", "nasa", "apollo",
        "artemis", "voyager", "rover", "mars mission", "space mission",
        "launch", "rocket", "lander", "orbiter", "probe", "iss",
        "international space station", "spacex", "esa", "jaxa",
    ]

    TECH_KEYWORDS = [
        "telescope", "jwst", "james webb", "hubble", "satellite",
        "optics", "spectroscopy", "infrared", "launch vehicle",
        "thruster", "propulsion", "instrument", "sensor", "detector",
        "imaging", "radio telescope",
    ]

    # ==================== CATEGORY BOOSTING ====================
    ENABLE_CATEGORY_BOOST: bool = False  # Disabled â€” rag_pipeline.py uses RRF
    CATEGORY_BOOST_FACTORS: dict = {
        "Planetary Science": 1.0,
        "Stars & Astrophysics": 1.0,
        "Galaxies & Cosmology": 1.0,
        "Space Missions": 1.0,
        "Space Technology": 1.0,
    }

    # ==================== LLM GENERATION ====================
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 1024
    LLM_TOP_P: float = 0.9
    MAX_CONTEXT_LENGTH: int = 18000  # Characters sent to LLM context

    # ==================== LOGGING ====================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_RETRIEVAL: bool = os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true"

    # ------------------------------------------------------------------
    @classmethod
    def validate(cls) -> bool:
        """Ensure required configuration is present."""
        errors = []

        if not (cls.GE_API_KEY or cls.LLM_API_KEY):
            errors.append(
                "API key not configured. Set GE_API_KEY in .env."
            )
        if not cls.LLM_BASE_URL:
            errors.append("LLM_BASE_URL not configured. Set it in .env.")
        if not cls.LLM_MODEL:
            errors.append("LLM_MODEL not configured. Set it in .env.")
        if not cls.EMBEDDING_MODEL:
            errors.append("EMBEDDING_MODEL not configured. Set it in .env.")
        if not cls.KNOWLEDGE_BASE_DIR.exists():
            errors.append(
                f"Knowledge base directory not found: {cls.KNOWLEDGE_BASE_DIR}"
            )
        if cls.LLM_MODEL and cls.EMBEDDING_MODEL and cls.LLM_MODEL == cls.EMBEDDING_MODEL:
            errors.append(
                f"WARNING: LLM_MODEL and EMBEDDING_MODEL are the same ({cls.LLM_MODEL}). "
                "Ensure LLM_MODEL is a chat model, not an embeddings-only model."
            )
        if errors:
            raise ValueError("\n".join(errors))
        return True

    @classmethod
    def get_query_category(cls, query: str) -> str | None:
        """Classify a query into an astronomy domain category."""
        q = query.lower()
        if any(kw in q for kw in cls.MISSIONS_KEYWORDS):
            return "Space Missions"
        if any(kw in q for kw in cls.TECH_KEYWORDS):
            return "Space Technology"
        if any(kw in q for kw in cls.COSMOLOGY_KEYWORDS):
            return "Galaxies & Cosmology"
        if any(kw in q for kw in cls.ASTROPHYSICS_KEYWORDS):
            return "Stars & Astrophysics"
        if any(kw in q for kw in cls.PLANETARY_KEYWORDS):
            return "Planetary Science"
        return None


# ==================== BACKWARDS-COMPATIBLE MODULE-LEVEL EXPORTS ====================
KNOWLEDGE_BASE_DIR = Config.KNOWLEDGE_BASE_DIR
VECTORSTORE_DIR = Config.VECTORSTORE_DIR
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
LLM_MODEL = Config.LLM_MODEL
GE_API_KEY = Config.GE_API_KEY
LLM_API_KEY = Config.LLM_API_KEY
LLM_BASE_URL = Config.LLM_BASE_URL
OPENAI_BASE_URL = Config.OPENAI_BASE_URL
TOP_K = Config.TOP_K
SCORE_THRESHOLD = Config.SCORE_THRESHOLD
CHUNK_SIZE = Config.CHUNK_SIZE
CHUNK_MAX_TOKENS = Config.CHUNK_MAX_TOKENS
CHUNK_OVERLAP = Config.CHUNK_OVERLAP
EMBEDDING_BATCH_SIZE = Config.EMBEDDING_BATCH_SIZE
EMBEDDING_MAX_RETRIES = Config.EMBEDDING_MAX_RETRIES
EMBEDDING_RETRY_BASE_DELAY = Config.EMBEDDING_RETRY_BASE_DELAY
COLLECTION_NAME = Config.COLLECTION_NAME
VECTOR_TOP_K = Config.VECTOR_TOP_K
BM25_TOP_K = Config.BM25_TOP_K
HYBRID_TOP_K = Config.HYBRID_TOP_K
RERANK_TOP_K = Config.RERANK_TOP_K
MIN_CHUNK_LENGTH = Config.MIN_CHUNK_LENGTH
MIN_RERANK_SCORE = Config.MIN_RERANK_SCORE
RERANKER_MODEL = Config.RERANKER_MODEL
IMAGE_ANALYSIS_MODEL = Config.IMAGE_ANALYSIS_MODEL
ENABLE_IMAGE_ANALYSIS = Config.ENABLE_IMAGE_ANALYSIS
USE_POSTGRES = Config.USE_POSTGRES
DATABASE_URL = Config.DATABASE_URL
DATABASE_ECHO = Config.DATABASE_ECHO

# Validate on import (print warning but don't crash)
try:
    Config.validate()
except ValueError as e:
    import sys
    print(f"âš ï¸  Configuration validation warning:\n{e}", file=sys.stderr)
