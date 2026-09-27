"""
api.py

Enterprise Policy RAG
FastAPI Backend + Built-in Developer Console

Run locally:

    uvicorn api:app --host 0.0.0.0 --port 8080 --reload

Open:

    http://localhost:8080

Swagger:

    http://localhost:8080/docs
"""

# ============================================================
# IMPORTS
# ============================================================

import html
import logging
import time
import uuid

from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)

from pydantic import (
    BaseModel,
    Field,
)

from rag import create_rag


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
# GLOBAL APPLICATION STATE
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
        "Starting Enterprise Policy RAG..."
    )

    try:

        rag = create_rag()

        logger.info(
            "RAG engine initialized successfully."
        )

    except Exception:

        logger.exception(
            "RAG initialization failed."
        )

        raise

    yield

    logger.info(
        "Enterprise Policy RAG shutting down."
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Enterprise Policy RAG",

    description=(
        "Enterprise employee-policy "
        "Retrieval-Augmented Generation API"
    ),

    version="1.0.0",

    lifespan=lifespan,

    docs_url="/docs",

    redoc_url="/redoc",
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
# API MODELS
# ============================================================

class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        examples=[
            "What happens to my outstanding loan if I resign?"
        ],
    )

    top_k: Optional[int] = Field(
        default=10,
        ge=1,
        le=20,
    )


class SourceResponse(BaseModel):

    number: int

    score: float

    source: str

    section: str

    text: str


class AskResponse(BaseModel):

    request_id: str

    question: str

    answer: str

    generation_provider: Optional[str]

    generation_model: Optional[str]

    sources: List[SourceResponse]

    processing_time_ms: float


class RetrievalRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=20,
    )


# ============================================================
# REQUEST ID + TIMING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def request_metadata(
    request: Request,
    call_next,
):

    request_id = str(
        uuid.uuid4()
    )

    request.state.request_id = request_id

    started = time.perf_counter()

    try:

        response = await call_next(
            request
        )

    except Exception:

        logger.exception(
            "Unhandled request failure | "
            "request_id=%s",
            request_id,
        )

        raise

    duration_ms = (
        time.perf_counter() - started
    ) * 1000

    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Processing-Time-MS"
    ] = f"{duration_ms:.2f}"

    logger.info(
        "HTTP | %s %s | %s | %.2fms | %s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
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
        "Application error | request_id=%s",
        request_id,
    )

    return JSONResponse(

        status_code=500,

        content={
            "error": "Internal server error",
            "message": (
                "The service could not "
                "complete the request."
            ),
            "request_id": request_id,
        },
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    uptime = 0

    if application_started_at:

        uptime = round(
            time.time()
            - application_started_at,
            2,
        )

    return {

        "status":
            "healthy",

        "application":
            "enterprise-policy-RAG",

        "version":
            "1.0.0",

        "rag_initialized":
            rag is not None,

        "uptime_seconds":
            uptime,
    }


# ============================================================
# READINESS
# ============================================================

@app.get("/ready")
def ready():

    if rag is None:

        raise HTTPException(

            status_code=
                status.HTTP_503_SERVICE_UNAVAILABLE,

            detail=
                "RAG service is not ready.",
        )

    return {

        "status":
            "ready",

        "rag_initialized":
            True,
    }


# ============================================================
# ASK
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(
    payload: AskRequest,
    request: Request,
):

    if rag is None:

        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready.",
        )

    request_id = request.state.request_id

    question = payload.question.strip()

    started = time.perf_counter()

    try:

        result = rag.ask(

            question=question,

            top_k=payload.top_k,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:

        logger.exception(
            "RAG request failed | %s",
            request_id,
        )

        raise HTTPException(

            status_code=503,

            detail=(
                "AI service temporarily "
                "unavailable."
            ),
        )

    processing_time_ms = (
        time.perf_counter() - started
    ) * 1000

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

        for source in result.get(
            "sources",
            [],
        )
    ]

    return AskResponse(

        request_id=request_id,

        question=question,

        answer=result["answer"],

        generation_provider=result.get(
            "generation_provider"
        ),

        generation_model=result.get(
            "generation_model"
        ),

        sources=sources,

        processing_time_ms=round(
            processing_time_ms,
            2,
        ),
    )


# ============================================================
# RETRIEVAL DEBUG
# ============================================================

@app.post("/debug/retrieval")
def debug_retrieval(
    payload: RetrievalRequest,
):

    if rag is None:

        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready.",
        )

    chunks = rag.retriever.retrieve(

        question=payload.question,

        top_k=payload.top_k,
    )

    results = []

    for rank, chunk in enumerate(
        chunks,
        start=1,
    ):

        section = " > ".join(

            heading

            for heading in [

                chunk.heading_1,
                chunk.heading_2,
                chunk.heading_3,

            ]

            if heading
        )

        results.append({

            "rank":
                rank,

            "score":
                round(
                    chunk.score,
                    4,
                ),

            "source":
                chunk.source,

            "section":
                section,

            "text":
                chunk.text,
        })

    return {

        "question":
            payload.question,

        "results":
            results,
    }


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api/info")
def api_info():

    return {

        "application":
            "enterprise-policy-RAG",

        "version":
            "1.0.0",

        "status":
            "ready"
            if rag is not None
            else "starting",

        "endpoints": {

            "console":
                "/",

            "health":
                "/health",

            "readiness":
                "/ready",

            "ask":
                "/ask",

            "retrieval_debug":
                "/debug/retrieval",

            "swagger":
                "/docs",

            "redoc":
                "/redoc",
        },
    }


# ============================================================
# BUILT-IN DEVELOPER CONSOLE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def developer_console():

    """
    A lightweight developer console served directly by FastAPI.

    No Streamlit.
    No React.
    No Node.
    No separate frontend service.
    """

    return HTMLResponse(
        content="""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Enterprise Policy RAG</title>

<style>

/* ==========================================================
   BASE
   ========================================================== */

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    background: #0b0d10;

    color: #e8eaed;

    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


/* ==========================================================
   HEADER
   ========================================================== */

header {

    border-bottom:
        1px solid #242830;

    padding:
        22px 32px;

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    background:
        #111419;
}


.brand {

    font-size:
        21px;

    font-weight:
        700;

    letter-spacing:
        -0.3px;
}


.subtitle {

    color:
        #8f98a7;

    font-size:
        13px;

    margin-top:
        5px;
}


.header-links a {

    color:
        #b8c0cc;

    text-decoration:
        none;

    margin-left:
        22px;

    font-size:
        14px;
}


.header-links a:hover {

    color:
        white;
}


/* ==========================================================
   CONTAINER
   ========================================================== */

.container {

    max-width:
        1450px;

    margin:
        auto;

    padding:
        28px;
}


/* ==========================================================
   STATUS CARDS
   ========================================================== */

.status-grid {

    display:
        grid;

    grid-template-columns:
        repeat(
            4,
            1fr
        );

    gap:
        14px;

    margin-bottom:
        25px;
}


.status-card {

    background:
        #12161c;

    border:
        1px solid #252b34;

    border-radius:
        10px;

    padding:
        17px;
}


.status-label {

    color:
        #8d96a4;

    font-size:
        12px;

    text-transform:
        uppercase;

    letter-spacing:
        .8px;
}


.status-value {

    margin-top:
        9px;

    font-size:
        16px;

    font-weight:
        600;
}


.good {

    color:
        #58d68d;
}


.bad {

    color:
        #ff6b6b;
}


/* ==========================================================
   MAIN GRID
   ========================================================== */

.main-grid {

    display:
        grid;

    grid-template-columns:
        1fr 1fr;

    gap:
        20px;
}


.panel {

    background:
        #12161c;

    border:
        1px solid #252b34;

    border-radius:
        12px;

    overflow:
        hidden;
}


.panel-header {

    padding:
        16px 18px;

    border-bottom:
        1px solid #252b34;

    font-weight:
        600;
}


.panel-body {

    padding:
        18px;
}


/* ==========================================================
   ENDPOINT SELECTOR
   ========================================================== */

.endpoint-row {

    display:
        flex;

    gap:
        10px;

    margin-bottom:
        18px;
}


.method {

    background:
        #174e36;

    color:
        #7ee2ad;

    border:
        1px solid #26694b;

    padding:
        9px 12px;

    border-radius:
        6px;

    font-size:
        13px;

    font-weight:
        700;
}


.endpoint {

    flex:
        1;

    background:
        #0d1014;

    border:
        1px solid #292f39;

    color:
        #e6e8eb;

    padding:
        9px 12px;

    border-radius:
        6px;

    font-family:
        monospace;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

button {

    border:
        0;

    border-radius:
        7px;

    padding:
        10px 16px;

    font-weight:
        600;

    cursor:
        pointer;
}


.primary {

    background:
        #e8eaed;

    color:
        #111;
}


.secondary {

    background:
        #20252d;

    color:
        #e8eaed;

    border:
        1px solid #343b46;
}


.buttons {

    display:
        flex;

    gap:
        10px;

    margin-top:
        14px;
}


/* ==========================================================
   CODE AREAS
   ========================================================== */

textarea {

    width:
        100%;

    min-height:
        230px;

    resize:
        vertical;

    background:
        #090b0e;

    color:
        #dce1e8;

    border:
        1px solid #292f39;

    border-radius:
        8px;

    padding:
        14px;

    font-family:
        "SFMono-Regular",
        Consolas,
        monospace;

    font-size:
        13px;

    line-height:
        1.55;
}


pre {

    margin:
        0;

    min-height:
        410px;

    max-height:
        680px;

    overflow:
        auto;

    background:
        #090b0e;

    border:
        1px solid #292f39;

    border-radius:
        8px;

    padding:
        15px;

    color:
        #cdd5df;

    font-size:
        13px;

    line-height:
        1.55;
}


/* ==========================================================
   PRESETS
   ========================================================== */

.presets {

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        8px;

    margin-bottom:
        15px;
}


.preset {

    background:
        #1b2027;

    color:
        #aeb7c3;

    border:
        1px solid #303743;

    padding:
        7px 10px;

    font-size:
        12px;
}


.preset:hover {

    background:
        #262d37;

    color:
        white;
}


/* ==========================================================
   RESPONSE META
   ========================================================== */

.response-meta {

    display:
        flex;

    gap:
        18px;

    margin-bottom:
        12px;

    color:
        #8993a1;

    font-size:
        12px;
}


/* ==========================================================
   CURL
   ========================================================== */

.curl-box {

    margin-top:
        22px;
}


.curl-title {

    font-size:
        13px;

    color:
        #9099a7;

    margin-bottom:
        8px;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (
    max-width: 900px
) {

    .main-grid {

        grid-template-columns:
            1fr;
    }

    .status-grid {

        grid-template-columns:
            1fr 1fr;
    }
}

</style>

</head>


<body>


<header>

<div>

<div class="brand">
enterprise-policy-RAG
</div>

<div class="subtitle">
FastAPI · Gemini Embeddings · Qdrant · Gemini/Groq Generation
</div>

</div>


<div class="header-links">

<a href="/docs">
Swagger
</a>

<a href="/redoc">
ReDoc
</a>

<a href="/health">
Health JSON
</a>

</div>

</header>


<div class="container">


<!-- =======================================================
     STATUS
     ======================================================= -->

<div class="status-grid">


<div class="status-card">

<div class="status-label">
API
</div>

<div
    class="status-value"
    id="apiStatus"
>
Checking...
</div>

</div>


<div class="status-card">

<div class="status-label">
RAG Engine
</div>

<div
    class="status-value"
    id="ragStatus"
>
Checking...
</div>

</div>


<div class="status-card">

<div class="status-label">
Readiness
</div>

<div
    class="status-value"
    id="readyStatus"
>
Checking...
</div>

</div>


<div class="status-card">

<div class="status-label">
Uptime
</div>

<div
    class="status-value"
    id="uptime"
>
--
</div>

</div>


</div>


<!-- =======================================================
     MAIN
     ======================================================= -->

<div class="main-grid">


<!-- REQUEST ================================================= -->

<div class="panel">

<div class="panel-header">
API Request
</div>


<div class="panel-body">


<div class="endpoint-row">

<select
    id="endpoint"
    class="endpoint"
    onchange="changeEndpoint()"
>

<option value="/ask">
POST /ask
</option>

<option value="/debug/retrieval">
POST /debug/retrieval
</option>

<option value="/health">
GET /health
</option>

<option value="/ready">
GET /ready
</option>

<option value="/api/info">
GET /api/info
</option>

</select>

</div>


<div class="presets">

<button
    class="preset"
    onclick="presetLoan()"
>
Loan + resignation
</button>

<button
    class="preset"
    onclick="presetTravel()"
>
International travel
</button>

<button
    class="preset"
    onclick="presetCertification()"
>
Certification
</button>

<button
    class="preset"
    onclick="presetUnknown()"
>
Unknown policy
</button>

</div>


<textarea
    id="payload"
>{
  "question": "What happens to my outstanding loan if I resign?",
  "top_k": 10
}</textarea>


<div class="buttons">

<button
    class="primary"
    onclick="sendRequest()"
>
Send Request
</button>


<button
    class="secondary"
    onclick="copyPayload()"
>
Copy Payload
</button>


<button
    class="secondary"
    onclick="copyCurl()"
>
Copy cURL
</button>

</div>


<div class="curl-box">

<div class="curl-title">
Generated cURL
</div>

<textarea
    id="curl"
    readonly
></textarea>

</div>


</div>

</div>


<!-- RESPONSE ================================================ -->

<div class="panel">

<div class="panel-header">
Live Response
</div>


<div class="panel-body">


<div class="response-meta">

<span id="httpStatus">
Status: --
</span>

<span id="responseTime">
Time: --
</span>

<span id="requestId">
Request: --
</span>

</div>


<pre id="response">Select an endpoint and click "Send Request".</pre>


<div class="buttons">

<button
    class="secondary"
    onclick="copyResponse()"
>
Copy Response
</button>

</div>


</div>

</div>


</div>

</div>


<script>

/* ==========================================================
   HELPERS
   ========================================================== */

const payload = document.getElementById(
    "payload"
);

const endpoint = document.getElementById(
    "endpoint"
);

const responseBox = document.getElementById(
    "response"
);


/* ==========================================================
   STATUS
   ========================================================== */

async function refreshStatus() {

    try {

        const healthResponse = await fetch(
            "/health"
        );

        const health = await healthResponse.json();


        document.getElementById(
            "apiStatus"
        ).innerHTML =
            '<span class="good">ONLINE</span>';


        document.getElementById(
            "ragStatus"
        ).innerHTML =
            health.rag_initialized
                ? '<span class="good">INITIALIZED</span>'
                : '<span class="bad">NOT READY</span>';


        document.getElementById(
            "uptime"
        ).textContent =
            health.uptime_seconds + " sec";


        const readyResponse = await fetch(
            "/ready"
        );


        document.getElementById(
            "readyStatus"
        ).innerHTML =
            readyResponse.ok
                ? '<span class="good">READY</span>'
                : '<span class="bad">NOT READY</span>';

    }

    catch {

        document.getElementById(
            "apiStatus"
        ).innerHTML =
            '<span class="bad">OFFLINE</span>';
    }
}


refreshStatus();

setInterval(
    refreshStatus,
    10000
);


/* ==========================================================
   PRESETS
   ========================================================== */

function setQuestion(question) {

    payload.value = JSON.stringify(
        {
            question: question,
            top_k: 10
        },
        null,
        2
    );

    endpoint.value = "/ask";

    updateCurl();
}


function presetLoan() {

    setQuestion(
        "What happens to my outstanding loan if I resign?"
    );
}


function presetTravel() {

    setQuestion(
        "What approvals do I need for international business travel?"
    );
}


function presetCertification() {

    setQuestion(
        "Can the company pay for my professional certification?"
    );
}


function presetUnknown() {

    setQuestion(
        "What company car will a new employee receive?"
    );
}


/* ==========================================================
   ENDPOINT CHANGE
   ========================================================== */

function changeEndpoint() {

    const value = endpoint.value;


    if (
        value === "/health"
        ||
        value === "/ready"
        ||
        value === "/api/info"
    ) {

        payload.disabled = true;

    }

    else {

        payload.disabled = false;
    }


    updateCurl();
}


/* ==========================================================
   CURL GENERATOR
   ========================================================== */

function updateCurl() {

    const path = endpoint.value;

    const base =
        window.location.origin;


    if (
        path === "/health"
        ||
        path === "/ready"
        ||
        path === "/api/info"
    ) {

        document.getElementById(
            "curl"
        ).value =
            `curl "${base}${path}"`;

        return;
    }


    document.getElementById(
        "curl"
    ).value =
`curl -X POST "${base}${path}" \\
  -H "Content-Type: application/json" \\
  -d '${payload.value.replace(
      /'/g,
      "'\\\\''"
  )}'`;
}


payload.addEventListener(
    "input",
    updateCurl
);


updateCurl();


/* ==========================================================
   SEND REQUEST
   ========================================================== */

async function sendRequest() {

    const path = endpoint.value;

    const started =
        performance.now();


    responseBox.textContent =
        "Loading...";


    document.getElementById(
        "httpStatus"
    ).textContent =
        "Status: --";


    document.getElementById(
        "requestId"
    ).textContent =
        "Request: --";


    try {

        let options = {};


        if (
            path !== "/health"
            &&
            path !== "/ready"
            &&
            path !== "/api/info"
        ) {

            let parsedPayload;


            try {

                parsedPayload =
                    JSON.parse(
                        payload.value
                    );

            }

            catch {

                throw new Error(
                    "Payload is not valid JSON."
                );
            }


            options = {

                method:
                    "POST",

                headers: {

                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify(
                        parsedPayload
                    )
            };
        }


        const result = await fetch(
            path,
            options
        );


        const duration =
            performance.now()
            - started;


        let data;


        try {

            data = await result.json();

        }

        catch {

            data = {
                error:
                    "Response was not JSON."
            };
        }


        responseBox.textContent =
            JSON.stringify(
                data,
                null,
                2
            );


        document.getElementById(
            "httpStatus"
        ).textContent =
            `Status: ${result.status}`;


        document.getElementById(
            "responseTime"
        ).textContent =
            `Time: ${duration.toFixed(0)} ms`;


        const requestId =
            result.headers.get(
                "X-Request-ID"
            );


        document.getElementById(
            "requestId"
        ).textContent =
            `Request: ${requestId || "--"}`;

    }

    catch (error) {

        responseBox.textContent =
            JSON.stringify(
                {
                    error:
                        error.message
                },
                null,
                2
            );
    }
}


/* ==========================================================
   COPY HELPERS
   ========================================================== */

async function copyPayload() {

    await navigator.clipboard.writeText(
        payload.value
    );
}


async function copyCurl() {

    await navigator.clipboard.writeText(

        document.getElementById(
            "curl"
        ).value
    );
}


async function copyResponse() {

    await navigator.clipboard.writeText(
        responseBox.textContent
    );
}


/* ==========================================================
   INITIALIZE
   ========================================================== */

changeEndpoint();

</script>


</body>

</html>
"""
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