"""
api.py

Enterprise Policy RAG API

Responsibilities:
- FastAPI application
- API contracts
- Rate limiting
- Request IDs
- Health/readiness
- RAG request handling
- Operational logging
- Developer console serving

The actual RAG implementation lives in rag.py.

Run locally:

    uvicorn api:app --host 0.0.0.0 --port 8080 --reload

Developer console:

    http://localhost:8080

Swagger:

    http://localhost:8080/docs
"""

# ============================================================
# IMPORTS
# ============================================================

import logging
import time
import uuid

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from pydantic import (
    BaseModel,
    Field,
)

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi import _rate_limit_exceeded_handler

from rag import create_rag


# ============================================================
# APPLICATION CONSTANTS
# ============================================================

APP_NAME = "enterprise-policy-RAG"

APP_VERSION = "1.1.0"

API_PREFIX = "/api/v1"

DEFAULT_TOP_K = 10

MAX_TOP_K = 15


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"

INDEX_FILE = STATIC_DIR / "index.html"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "enterprise-policy-rag-api"
)


# ============================================================
# RATE LIMITING
# ============================================================

limiter = Limiter(
    key_func=get_remote_address
)


# ============================================================
# APPLICATION STATE
# ============================================================

rag = None

application_started_at = None


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag
    global application_started_at

    application_started_at = time.time()

    logger.info(
        "APPLICATION_STARTING | "
        "name=%s | version=%s",
        APP_NAME,
        APP_VERSION,
    )

    try:

        rag = create_rag()

        logger.info(
            "RAG_INITIALIZED"
        )

    except Exception:

        logger.exception(
            "RAG_INITIALIZATION_FAILED"
        )

        raise

    yield

    logger.info(
        "APPLICATION_SHUTDOWN"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Enterprise Policy RAG API",

    description=(
        "Enterprise employee-policy "
        "Retrieval-Augmented Generation service."
    ),

    version=APP_VERSION,

    lifespan=lifespan,

    docs_url="/docs",

    redoc_url="/redoc",

    openapi_url="/openapi.json",
)


# ============================================================
# RATE LIMITER REGISTRATION
# ============================================================

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description=(
            "Employee policy question."
        ),
        examples=[
            (
                "What happens to my outstanding "
                "loan if I resign?"
            )
        ],
    )

    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=3,
        le=MAX_TOP_K,
        description=(
            "Number of policy chunks retrieved "
            "before generation."
        ),
    )

    role: str = Field(

        default="employee",

        pattern=
            "^(employee|manager|hr)$",

        description=(
            "Simulated access role for "
            "authorization-aware retrieval."
        ),
    )


class RetrievalRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description=(
            "Question used for retrieval testing."
        ),
    )

    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=3,
        le=MAX_TOP_K,
    )

    role: str = Field(
        default="employee",
        pattern="^(employee|manager|hr)$",
    )


class SourceResponse(BaseModel):

    number: int

    score: float

    source: str

    section: str

    text: str




class TimingResponse(BaseModel):

    cache_lookup_ms: float

    retrieval_ms: float

    context_ms: float

    generation_ms: float

    total_ms: float


class AskResponse(BaseModel):

    request_id: str

    question: str

    role: str

    knowledge_base_version: str

    answer: str

    generation_provider: Optional[str]

    generation_model: Optional[str]

    fallback_used: bool

    cache_hit: bool

    retrieval_top_score: Optional[float]

    timings: TimingResponse

    sources: List[SourceResponse]

    processing_time_ms: float

class RetrievalResult(BaseModel):

    rank: int

    score: float

    source: str

    section: str

    text: str


    

class RetrievalResponse(BaseModel):

    request_id: str

    question: str

    top_k: int

    retrieval_time_ms: float

    results: List[RetrievalResult]


class HealthResponse(BaseModel):

    status: str

    application: str

    version: str

    uptime_seconds: float


class ReadyResponse(BaseModel):

    status: str

    rag_initialized: bool


# ============================================================
# REQUEST ID + HTTP TIMING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def request_metadata_middleware(
    request: Request,
    call_next,
):

    request_id = str(
        uuid.uuid4()
    )

    request.state.request_id = (
        request_id
    )

    started = (
        time.perf_counter()
    )

    try:

        response = await call_next(
            request
        )

    except Exception:

        logger.exception(
            "HTTP_UNHANDLED_ERROR | "
            "request_id=%s | "
            "method=%s | "
            "path=%s",
            request_id,
            request.method,
            request.url.path,
        )

        raise

    duration_ms = (

        time.perf_counter()
        - started

    ) * 1000


    response.headers[
        "X-Request-ID"
    ] = request_id


    response.headers[
        "X-Processing-Time-MS"
    ] = f"{duration_ms:.2f}"


    logger.info(
        "HTTP_REQUEST | "
        "request_id=%s | "
        "method=%s | "
        "path=%s | "
        "status=%s | "
        "duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )


    return response


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    error: Exception,
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )


    logger.exception(
        "APPLICATION_ERROR | "
        "request_id=%s",
        request_id,
    )


    return JSONResponse(

        status_code=500,

        content={

            "error":
                "internal_server_error",

            "message":
                (
                    "The service could not "
                    "complete the request."
                ),

            "request_id":
                request_id,
        },
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def require_rag():

    if rag is None:

        raise HTTPException(

            status_code=
                status.HTTP_503_SERVICE_UNAVAILABLE,

            detail=
                "RAG service is not ready.",
        )


def build_section(
    chunk,
) -> str:

    return " > ".join(

        heading

        for heading in [

            chunk.heading_1,
            chunk.heading_2,
            chunk.heading_3,

        ]

        if heading
    )


# ============================================================
# DEVELOPER CONSOLE
# ============================================================

@app.get(
    "/",
    response_class=FileResponse,
    include_in_schema=False,
)
def developer_console():

    if not INDEX_FILE.exists():

        raise HTTPException(

            status_code=500,

            detail=(
                "Developer console file "
                "static/index.html was not found."
            ),
        )


    return FileResponse(
        INDEX_FILE
    )



# ============================================================
# RUNTIME METRICS
# ============================================================

@app.get(
    "/metrics",
    tags=["System"],
)
def metrics():

    require_rag()


    return rag.metrics.snapshot(

        cache_size=
            rag.cache.size()
    )

# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health():

    uptime = 0.0


    if application_started_at:

        uptime = round(

            time.time()
            - application_started_at,

            2,
        )


    return HealthResponse(

        status="healthy",

        application=APP_NAME,

        version=APP_VERSION,

        uptime_seconds=uptime,
    )


# ============================================================
# READINESS
# ============================================================

@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System"],
)
def readiness():

    require_rag()


    return ReadyResponse(

        status="ready",

        rag_initialized=True,
    )


# ============================================================
# API INFORMATION
# ============================================================

@app.get(
    "/api/info",
    tags=["System"],
)
def api_info():

    return {

        "application":
            APP_NAME,

        "version":
            APP_VERSION,

        "status":
            (
                "ready"
                if rag is not None
                else "starting"
            ),

        "api_version":
            "v1",

        "features": [

            "Gemini query embeddings",

            "Qdrant vector retrieval",

            "Gemini generation",

            "Gemini model fallback",

            "Groq provider fallback",

            "rate limiting",

            "request tracing",

            "retrieval observability",

            "Swagger/OpenAPI",

            "built-in developer console",
        ],

        "rate_limits": {

            "ask":
                "20 requests/minute/IP",

            "retrieval":
                "30 requests/minute/IP",
        },

        "endpoints": {

            "console":
                "/",

            "health":
                "/health",

            "readiness":
                "/ready",

            "ask":
                f"{API_PREFIX}/ask",

            "retrieval":
                f"{API_PREFIX}/retrieval",

            "swagger":
                "/docs",

            "redoc":
                "/redoc",
        },
    }


# ============================================================
# MAIN RAG ENDPOINT
# ============================================================

@app.post(
    f"{API_PREFIX}/ask",
    response_model=AskResponse,
    tags=["RAG"],
)
@limiter.limit("20/minute")
def ask_policy(
    request: Request,
    payload: AskRequest,
):

    require_rag()


    request_id = (
        request.state.request_id
    )


    question = (
        payload.question.strip()
    )


    started = (
        time.perf_counter()
    )


    logger.info(
        "RAG_REQUEST | "
        "request_id=%s | "
        "top_k=%d | "
        "question=%r",
        request_id,
        payload.top_k,
        question,
    )


    try:

        result = rag.ask(

            question=
                question,

            top_k=
                payload.top_k,

            role=
                payload.role,
        )


    except ValueError as error:

        raise HTTPException(

            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                str(error),
        )


    except Exception:

        logger.exception(
            "RAG_REQUEST_FAILED | "
            "request_id=%s",
            request_id,
        )


        raise HTTPException(

            status_code=
                status.HTTP_503_SERVICE_UNAVAILABLE,

            detail=(
                "The AI service is temporarily "
                "unavailable. Please try again."
            ),
        )


    processing_time_ms = (

        time.perf_counter()
        - started

    ) * 1000


    raw_sources = result.get(
        "sources",
        [],
    )


    sources = [

        SourceResponse(

            number=source.get(
                "number",
                0,
            ),

            score=source.get(
                "score",
                0.0,
            ),

            source=source.get(
                "source",
                "",
            ),

            section=source.get(
                "section",
                "",
            ),

            text=source.get(
                "text",
                "",
            ),
        )

        for source
        in raw_sources
    ]


    retrieval_top_score = None


    if raw_sources:

        retrieval_top_score = max(

            source.get(
                "score",
                0.0,
            )

            for source
            in raw_sources
        )


    retrieved_sections = [

        source.get(
            "section",
            "",
        )

        for source
        in raw_sources
    ]


    provider = result.get(
        "generation_provider"
    )


    model = result.get(
        "generation_model"
    )


    fallback_used = (
        provider == "groq"
    )


    logger.info(
        "RAG_COMPLETED | "
        "request_id=%s | "
        "provider=%s | "
        "model=%s | "
        "fallback_used=%s | "
        "retrieval_top_score=%s | "
        "retrieved_sections=%s | "
        "duration_ms=%.2f",
        request_id,
        provider,
        model,
        fallback_used,
        retrieval_top_score,
        retrieved_sections,
        processing_time_ms,
    )


    return AskResponse(

        request_id=
            request_id,

        question=
            question,

        answer=
            result["answer"],

        role=
            result.get(
                "role",
                payload.role,
            ),

        knowledge_base_version=
            result.get(
                "knowledge_base_version",
                "unknown",
            ),

        generation_provider=
            provider,

        generation_model=
            model,

        fallback_used=
            fallback_used,

        cache_hit=
            result.get(
                "cache_hit",
                False,
            ),

        retrieval_top_score=
            retrieval_top_score,

        timings=
            TimingResponse(
                **result.get(
                    "timings",
                    {
                        "cache_lookup_ms": 0.0,
                        "retrieval_ms": 0.0,
                        "context_ms": 0.0,
                        "generation_ms": 0.0,
                        "total_ms": processing_time_ms,
                    },
                )
            ),

        sources=
            sources,

        processing_time_ms=
            round(
                processing_time_ms,
                2,
            ),
    )


# ============================================================
# RETRIEVAL ENDPOINT
# ============================================================

@app.post(
    f"{API_PREFIX}/retrieval",
    response_model=RetrievalResponse,
    tags=["RAG"],
)
@limiter.limit("30/minute")
def retrieve_policy_chunks(
    request: Request,
    payload: RetrievalRequest,
):

    require_rag()


    request_id = (
        request.state.request_id
    )


    question = (
        payload.question.strip()
    )


    started = (
        time.perf_counter()
    )


    try:

        chunks = (
            rag.retriever.retrieve(

                question=question,

                top_k=payload.top_k,

                role=payload.role,
            )
        )


    except ValueError as error:

        raise HTTPException(

            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                str(error),
        )


    except Exception:

        logger.exception(
            "RETRIEVAL_FAILED | "
            "request_id=%s",
            request_id,
        )


        raise HTTPException(

            status_code=
                status.HTTP_503_SERVICE_UNAVAILABLE,

            detail=(
                "Retrieval service is temporarily "
                "unavailable."
            ),
        )


    retrieval_time_ms = (

        time.perf_counter()
        - started

    ) * 1000


    results = [

        RetrievalResult(

            rank=rank,

            score=round(
                chunk.score,
                4,
            ),

            source=
                chunk.source,

            section=
                build_section(
                    chunk
                ),

            text=
                chunk.text,
        )

        for rank, chunk in enumerate(
            chunks,
            start=1,
        )
    ]


    logger.info(
        "RETRIEVAL_COMPLETED | "
        "request_id=%s | "
        "top_k=%d | "
        "top_score=%s | "
        "duration_ms=%.2f",
        request_id,
        payload.top_k,
        (
            results[0].score
            if results
            else None
        ),
        retrieval_time_ms,
    )


    return RetrievalResponse(

        request_id=
            request_id,

        question=
            question,

        top_k=
            payload.top_k,

        retrieval_time_ms=
            round(
                retrieval_time_ms,
                2,
            ),

        results=
            results,
    )


# ============================================================
# LOCAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        "api:app",

        host="0.0.0.0",

        port=8080,

        reload=True,
    )