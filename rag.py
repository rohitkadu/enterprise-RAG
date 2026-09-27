"""
rag.py

Enterprise Policy RAG Engine

Architecture:

    User Question
         |
         v
    Gemini Embedding
         |
         v
    Qdrant Vector Search
         |
         v
    Top-K Policy Chunks
         |
         v
    Context Builder
         |
         v
    Gemini Generation
         |
         | temporary failure
         v
    Gemini Model Fallbacks
         |
         | all Gemini models fail
         v
    Groq Fallback
         |
         v
       Answer

The existing document embeddings already live in Qdrant.

This file performs QUERY-TIME RAG only.
It does NOT ingest or re-embed the handbook.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import time
import logging

import threading
from copy import deepcopy

from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Dict,
    Any,
)

from dotenv import load_dotenv

from google import genai

from groq import Groq

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)


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
    "enterprise-policy-rag"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL"
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)


# ============================================================
# VALIDATE ENVIRONMENT VARIABLES
# ============================================================

required_variables = {

    "GEMINI_API_KEY":
        GEMINI_API_KEY,

    "GROQ_API_KEY":
        GROQ_API_KEY,

    "QDRANT_URL":
        QDRANT_URL,

    "QDRANT_API_KEY":
        QDRANT_API_KEY,
}


for variable_name, value in (
    required_variables.items()
):

    if not value:

        raise RuntimeError(
            f"Missing environment variable: "
            f"{variable_name}"
        )


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
#
# Keep using the SAME embedding model that was used when
# your document chunks were originally inserted into Qdrant.

EMBEDDING_MODEL = (
    "gemini-embedding-2"
)


# ------------------------------------------------------------
# GEMINI GENERATION FALLBACK CHAIN
# ------------------------------------------------------------
#
# IMPORTANT:
#
# Keep only model IDs that are actually available to your
# Gemini project.
#
# The router tries them in this order.

GEMINI_GENERATION_MODELS = [

    "gemini-3.8-flash",

    "gemini-3.7-flash",

    "gemini-3.6-flash",

    "gemini-3.5-flash",
]


# ------------------------------------------------------------
# GROQ FINAL FALLBACK
# ------------------------------------------------------------

GROQ_GENERATION_MODELS = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b"
)


CIRCUIT_BREAKER_SECONDS = 120
QUOTA_CIRCUIT_BREAKER_SECONDS = 1800


# ------------------------------------------------------------
# QDRANT COLLECTION
# ------------------------------------------------------------

COLLECTION_NAME = (
    "acme_employee_policies_v2"
)


# ============================================================
# KNOWLEDGE BASE VERSION
# ============================================================

# IMPORTANT:
#
# Increment this whenever the indexed policy knowledge changes.
#
# Examples:
#
# 2.0 -> current policy corpus
# 2.1 -> policies updated
# 3.0 -> major corpus rebuild
#
# This version becomes part of the cache key, which prevents
# stale cached answers after a knowledge-base update.

KNOWLEDGE_BASE_VERSION = "2.0"


# ============================================================
# ACCESS CONTROL
# ============================================================

ALLOWED_ROLES = {
    "employee",
    "manager",
    "hr",
}


DEFAULT_ROLE = "employee"

# ------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------

TOP_K = 10


# ============================================================
# RAG CACHE CONFIGURATION
# ============================================================

# Cache complete RAG responses for 10 minutes.
#
# This is intentionally an in-memory cache for now:
#
# - zero infrastructure
# - very fast
# - good for one Render instance
#
# Later, Redis can replace this without changing the RAG flow.

CACHE_TTL_SECONDS = 600

CACHE_MAX_ENTRIES = 500


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class RetrievedChunk:

    chunk_id: str

    text: str

    score: float

    source: str

    heading_1: Optional[str] = None

    heading_2: Optional[str] = None

    heading_3: Optional[str] = None


# ============================================================
# RAG PROMPT
# ============================================================

def build_rag_prompt(
    question: str,
    context: str,
) -> str:

    """
    Create one common prompt.

    Gemini and Groq receive exactly the same retrieved context
    and instructions.
    """

    return f"""
You are the employee-policy assistant for Acme Global Services.

Your job is to answer employee questions using ONLY the supplied
policy excerpts.

STRICT RULES:

1. Use only information contained in the supplied context.

2. Do not invent policies, limits, eligibility rules, approvals,
   exceptions, benefits, or procedures.

3. If the context does not contain enough information to answer
   the question, explicitly say:

   "I don't know based on the available policy documents."

4. If multiple policy excerpts are relevant, combine them carefully.

5. If policies contain conditions, exceptions, approval requirements,
   monetary limits, time limits, or eligibility criteria, mention them.

6. When making a factual policy statement, cite the relevant source
   using [Source N].

7. Do not treat general world knowledge as company policy.

8. If the question is ambiguous, explain the ambiguity rather than
   inventing an interpretation.

9. Prefer concise and clear answers.

10. Do not mention retrieval scores in the answer.


POLICY CONTEXT:

{context}


EMPLOYEE QUESTION:

{question}


ANSWER:
""".strip()





# ============================================================
# GEMINI EMBEDDING SERVICE
# ============================================================

class GeminiEmbeddingService:

    """
    Generates query embeddings using the same embedding model
    used when the policy corpus was indexed.
    """

    def __init__(
        self,
        client,
        model: str,
        retries: int = 3,
    ):

        self.client = client

        self.model = model

        self.retries = retries


    def embed_text(
        self,
        text: str,
    ) -> List[float]:

        if not text.strip():

            raise ValueError(
                "Cannot embed empty text."
            )


        last_error = None


        for attempt in range(
            1,
            self.retries + 1,
        ):

            try:

                logger.info(
                    "Embedding query | "
                    "model=%s | "
                    "attempt=%d/%d",
                    self.model,
                    attempt,
                    self.retries,
                )


                response = (
                    self.client.models.embed_content(

                        model=
                            self.model,

                        contents=
                            text,
                    )
                )


                vector = list(

                    response
                    .embeddings[0]
                    .values
                )


                logger.info(
                    "Query embedding generated | "
                    "dimensions=%d",
                    len(vector),
                )


                return vector


            except Exception as error:

                last_error = error


                status_code = getattr(
                    error,
                    "status_code",
                    None,
                )


                # google-genai does not always expose
                # status_code consistently.

                if status_code is None:

                    error_text = str(
                        error
                    ).upper()


                    if (
                        "429" in error_text
                        or "RESOURCE_EXHAUSTED"
                        in error_text
                    ):

                        status_code = 429


                    elif (
                        "503" in error_text
                        or "UNAVAILABLE"
                        in error_text
                    ):

                        status_code = 503


                if status_code not in {

                    429,
                    500,
                    502,
                    503,
                    504,

                }:

                    raise


                if attempt < self.retries:

                    delay = (

                        1.5

                        * (

                            2

                            ** (
                                attempt - 1
                            )
                        )
                    )


                    logger.warning(
                        "Embedding temporarily failed | "
                        "status=%s | "
                        "retrying in %.1fs",
                        status_code,
                        delay,
                    )


                    time.sleep(
                        delay
                    )


        raise RuntimeError(
            "Gemini embedding failed "
            "after all retries."
        ) from last_error

# ============================================================
# QDRANT VECTOR STORE
# ============================================================

class QdrantVectorStore:

    """
    Searches the existing Qdrant collection.

    No ingestion occurs here.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
    ):

        self.client = client

        self.collection_name = (
            collection_name
        )


    def search(
        self,
        query_vector: List[float],
        top_k: int,
        role: str,
    ) -> List[RetrievedChunk]:

        """
        Search Qdrant with authorization filtering.

        SECURITY PRINCIPLE:

            authorization happens BEFORE documents
            reach the LLM.

        Qdrant therefore returns only chunks whose
        access_roles contain the caller's role.
        """

        logger.info(
            "Searching Qdrant | "
            "collection=%s | "
            "top_k=%d | "
            "role=%s",
            self.collection_name,
            top_k,
            role,
        )


        access_filter = Filter(

            must=[

                FieldCondition(

                    key="access_roles",

                    match=MatchValue(
                        value=role
                    ),
                )

            ]
        )


        response = (
            self.client.query_points(

                collection_name=
                    self.collection_name,

                query=
                    query_vector,

                query_filter=
                    access_filter,

                limit=
                    top_k,

                with_payload=
                    True,
            )
        )


        results = []


        for point in response.points:

            payload = (
                point.payload
                or {}
            )


            results.append(

                RetrievedChunk(

                    chunk_id=payload.get(
                        "chunk_id",
                        str(point.id),
                    ),

                    text=payload.get(
                        "text",
                        "",
                    ),

                    score=float(
                        point.score
                    ),

                    source=payload.get(
                        "source",
                        "",
                    ),

                    heading_1=payload.get(
                        "heading_1"
                    ),

                    heading_2=payload.get(
                        "heading_2"
                    ),

                    heading_3=payload.get(
                        "heading_3"
                    ),
                )
            )


        logger.info(
            "Qdrant returned %d authorized chunks | "
            "role=%s",
            len(results),
            role,
        )


        return results

# ============================================================
# RETRIEVER
# ============================================================

class Retriever:

    def __init__(
        self,
        embedding_service:
            GeminiEmbeddingService,
        vector_store:
            QdrantVectorStore,
        default_top_k: int,
    ):

        self.embedding_service = (
            embedding_service
        )

        self.vector_store = (
            vector_store
        )

        self.default_top_k = (
            default_top_k
        )


    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        role: str = DEFAULT_ROLE,
    ) -> List[RetrievedChunk]:

        question = (
            question.strip()
        )


        role = (
            role
            .strip()
            .lower()
        )


        if not question:

            raise ValueError(
                "Question cannot be empty."
            )


        if role not in ALLOWED_ROLES:

            raise ValueError(
                f"Unsupported role: {role}"
            )


        # ========================================================
        # QUERY EMBEDDING
        # ========================================================

        query_vector = (
            self.embedding_service
            .embed_text(
                question
            )
        )


        # ========================================================
        # AUTHORIZED VECTOR SEARCH
        # ========================================================

        return self.vector_store.search(

            query_vector=
                query_vector,

            top_k=
                top_k
                or self.default_top_k,

            role=
                role,
        )

# ============================================================
# CONTEXT BUILDER
# ============================================================

class ContextBuilder:

    """
    Converts retrieved chunks into structured context for
    the generation model.
    """

    @staticmethod
    def build(
        chunks: List[RetrievedChunk],
    ) -> str:

        context_blocks = []


        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            heading_path = " > ".join(

                heading

                for heading in [

                    chunk.heading_1,

                    chunk.heading_2,

                    chunk.heading_3,

                ]

                if heading
            )


            block = f"""
[SOURCE {index}]

Document:
{chunk.source}

Section:
{heading_path or "Unknown"}

Content:
{chunk.text}
""".strip()


            context_blocks.append(
                block
            )


        return "\n\n---\n\n".join(
            context_blocks
        )





# ============================================================
# MODEL CIRCUIT BREAKER
# ============================================================

class ModelCircuitBreaker:

    """
    Thread-safe per-model circuit breaker.

    Example:

        gemini-3.8
             ↓
            429
             ↓
        circuit OPEN
             ↓
        future requests skip 3.8 temporarily

    Each model has its own independent circuit.
    """

    def __init__(self):

        self._open_until = {}

        self._lock = threading.Lock()


    def is_available(
        self,
        model: str,
    ) -> bool:

        with self._lock:

            open_until = self._open_until.get(
                model
            )


            if open_until is None:

                return True


            if time.time() >= open_until:

                self._open_until.pop(
                    model,
                    None,
                )

                logger.info(
                    "CIRCUIT_CLOSED | model=%s",
                    model,
                )

                return True


            return False


    def open(
        self,
        model: str,
        seconds: int,
        reason: str,
    ):

        with self._lock:

            self._open_until[
                model
            ] = (
                time.time()
                + seconds
            )


        logger.warning(
            "CIRCUIT_OPEN | "
            "model=%s | "
            "seconds=%d | "
            "reason=%s",
            model,
            seconds,
            reason,
        )


    def remaining_seconds(
        self,
        model: str,
    ) -> int:

        with self._lock:

            open_until = self._open_until.get(
                model
            )


            if open_until is None:

                return 0


            return max(
                0,
                int(
                    open_until
                    - time.time()
                ),
            )



# ============================================================
# GEMINI GENERATION SERVICE
# ============================================================

class GeminiGenerationService:

    """
    Gemini primary generation service.

    Important behavior:

    NORMAL:
        try Gemini models in priority order.

    QUOTA EXHAUSTED:
        open circuit for that model.

    TEMPORARY FAILURE:
        open shorter circuit.

    OPEN CIRCUIT:
        skip model immediately.

    This prevents every user request from repeatedly waiting
    for models that we already know are unavailable.
    """

    def __init__(
        self,
        client,
        models: List[str],
        circuit_breaker: ModelCircuitBreaker,
    ):

        self.client = client

        self.models = models

        self.circuit_breaker = (
            circuit_breaker
        )


    # ========================================================
    # STATUS CODE
    # ========================================================

    @staticmethod
    def _get_status_code(
        error: Exception,
    ):

        status_code = getattr(
            error,
            "status_code",
            None,
        )


        if status_code:

            return status_code


        # google-genai exceptions do not always expose
        # status_code consistently, as your logs demonstrated.

        text = str(
            error
        ).upper()


        if (
            "429"
            in text
            or "RESOURCE_EXHAUSTED"
            in text
        ):

            return 429


        if (
            "503"
            in text
            or "UNAVAILABLE"
            in text
        ):

            return 503


        if "500" in text:

            return 500


        if "502" in text:

            return 502


        if "504" in text:

            return 504


        return None


    # ========================================================
    # QUOTA FAILURE
    # ========================================================

    @staticmethod
    def _is_quota_failure(
        error: Exception,
    ) -> bool:

        text = str(
            error
        ).lower()


        return (
            "resource_exhausted"
            in text

            or "quota exceeded"
            in text

            or "free_tier_requests"
            in text
        )


    # ========================================================
    # SINGLE MODEL
    # ========================================================

    def _generate_with_model(
        self,
        model: str,
        prompt: str,
    ) -> str:

        response = (
            self.client.models
            .generate_content(

                model=model,

                contents=prompt,
            )
        )


        answer = response.text


        if not answer:

            raise RuntimeError(
                f"Gemini model {model} "
                f"returned an empty response."
            )


        return answer


    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        question: str,
        context: str,
    ) -> Dict[str, str]:

        prompt = build_rag_prompt(

            question=question,

            context=context,
        )


        attempted_models = []


        for model in self.models:

            # =================================================
            # CIRCUIT CHECK
            # =================================================

            if not self.circuit_breaker.is_available(
                model
            ):

                remaining = (
                    self.circuit_breaker
                    .remaining_seconds(
                        model
                    )
                )


                logger.info(
                    "GENERATION_MODEL_SKIPPED | "
                    "provider=gemini | "
                    "model=%s | "
                    "circuit_open=true | "
                    "remaining_seconds=%d",
                    model,
                    remaining,
                )


                continue


            attempted_models.append(
                model
            )


            logger.info(
                "GENERATION_ATTEMPT | "
                "provider=gemini | "
                "model=%s",
                model,
            )


            try:

                answer = (
                    self._generate_with_model(

                        model=model,

                        prompt=prompt,
                    )
                )


                logger.info(
                    "GENERATION_SUCCESS | "
                    "provider=gemini | "
                    "model=%s",
                    model,
                )


                return {

                    "answer":
                        answer,

                    "provider":
                        "gemini",

                    "model":
                        model,
                }


            except Exception as error:

                status_code = (
                    self._get_status_code(
                        error
                    )
                )


                # =================================================
                # QUOTA EXHAUSTED
                # =================================================

                if self._is_quota_failure(
                    error
                ):

                    self.circuit_breaker.open(

                        model=model,

                        seconds=
                            QUOTA_CIRCUIT_BREAKER_SECONDS,

                        reason=
                            "quota_exhausted",
                    )


                    logger.warning(
                        "GEMINI_QUOTA_EXHAUSTED | "
                        "model=%s | "
                        "moving_to_next_model",
                        model,
                    )


                    continue


                # =================================================
                # TEMPORARY SERVICE FAILURE
                # =================================================

                if status_code in {

                    500,
                    502,
                    503,
                    504,

                }:

                    self.circuit_breaker.open(

                        model=model,

                        seconds=
                            CIRCUIT_BREAKER_SECONDS,

                        reason=
                            f"http_{status_code}",
                    )


                    logger.warning(
                        "GEMINI_TEMPORARY_FAILURE | "
                        "model=%s | "
                        "status=%s",
                        model,
                        status_code,
                    )


                    continue


                # =================================================
                # OTHER MODEL FAILURE
                #
                # Don't make the entire service unavailable merely
                # because one configured model rejected a request.
                # =================================================

                logger.warning(
                    "GEMINI_MODEL_FAILURE | "
                    "model=%s | "
                    "status=%s | "
                    "error=%s",
                    model,
                    status_code,
                    error,
                )


                continue


        raise RuntimeError(

            "No Gemini generation model was available. "
            f"Attempted={attempted_models}"
        )

# ============================================================
# GROQ GENERATION SERVICE
# ============================================================

class GroqGenerationService:

    """
    Multi-model Groq fallback chain.

    Priority:

        GPT-OSS 120B
              ↓
        Qwen 3.8 27B
              ↓
        GPT-OSS 20B

    Every model also has circuit-breaker protection.
    """

    def __init__(
        self,
        client: Groq,
        models: List[str],
        circuit_breaker: ModelCircuitBreaker,
    ):

        self.client = client

        self.models = models

        self.circuit_breaker = (
            circuit_breaker
        )


    @staticmethod
    def _status_code(
        error: Exception,
    ):

        status_code = getattr(
            error,
            "status_code",
            None,
        )


        if status_code:

            return status_code


        text = str(
            error
        )


        if "429" in text:

            return 429


        if "503" in text:

            return 503


        if "502" in text:

            return 502


        if "500" in text:

            return 500


        return None


    def generate(
        self,
        question: str,
        context: str,
    ) -> Dict[str, str]:

        prompt = build_rag_prompt(

            question=question,

            context=context,
        )


        last_error = None


        for model in self.models:

            # =================================================
            # CIRCUIT CHECK
            # =================================================

            if not self.circuit_breaker.is_available(
                model
            ):

                logger.info(
                    "GENERATION_MODEL_SKIPPED | "
                    "provider=groq | "
                    "model=%s | "
                    "circuit_open=true",
                    model,
                )


                continue


            logger.info(
                "GENERATION_ATTEMPT | "
                "provider=groq | "
                "model=%s",
                model,
            )


            try:

                response = (
                    self.client
                    .chat
                    .completions
                    .create(

                        model=
                            model,

                        messages=[

                            {
                                "role":
                                    "user",

                                "content":
                                    prompt,
                            }

                        ],

                        temperature=
                            0.1,
                    )
                )


                answer = (

                    response
                    .choices[0]
                    .message
                    .content
                )


                if not answer:

                    raise RuntimeError(
                        f"Groq model {model} "
                        f"returned an empty response."
                    )


                logger.info(
                    "GENERATION_SUCCESS | "
                    "provider=groq | "
                    "model=%s",
                    model,
                )


                return {

                    "answer":
                        answer,

                    "provider":
                        "groq",

                    "model":
                        model,
                }


            except Exception as error:

                last_error = error


                status_code = (
                    self._status_code(
                        error
                    )
                )


                # =================================================
                # RATE LIMIT
                # =================================================

                if status_code == 429:

                    self.circuit_breaker.open(

                        model=model,

                        seconds=
                            CIRCUIT_BREAKER_SECONDS,

                        reason=
                            "rate_limit",
                    )


                # =================================================
                # TEMPORARY SERVICE FAILURE
                # =================================================

                elif status_code in {

                    500,
                    502,
                    503,

                }:

                    self.circuit_breaker.open(

                        model=model,

                        seconds=
                            CIRCUIT_BREAKER_SECONDS,

                        reason=
                            f"http_{status_code}",
                    )


                logger.warning(
                    "GROQ_MODEL_FAILURE | "
                    "model=%s | "
                    "status=%s | "
                    "error=%s",
                    model,
                    status_code,
                    error,
                )


                # Immediately try next Groq model.

                continue


        raise RuntimeError(
            "All Groq generation models failed."
        ) from last_error

# ============================================================
# GENERATION ROUTER
# ============================================================

class GenerationRouter:

    """
    Provider-level generation routing.

    Flow:

        Gemini chain
             |
          success
             |
           answer

    OR

        Gemini unavailable
             |
             v
        Groq chain
             |
          success
             |
           answer
    """

    def __init__(
        self,
        gemini_generator:
            GeminiGenerationService,
        groq_generator:
            GroqGenerationService,
    ):

        self.gemini_generator = (
            gemini_generator
        )

        self.groq_generator = (
            groq_generator
        )


    def generate(
        self,
        question: str,
        context: str,
    ) -> Dict[str, str]:

        # ====================================================
        # PRIMARY PROVIDER
        # ====================================================

        try:

            return (
                self.gemini_generator
                .generate(

                    question=question,

                    context=context,
                )
            )


        except Exception as error:

            logger.warning(
                "GENERATION_PROVIDER_FALLBACK | "
                "from=gemini | "
                "to=groq | "
                "reason=%s",
                error,
            )


        # ====================================================
        # SECONDARY PROVIDER
        # ====================================================

        return (
            self.groq_generator
            .generate(

                question=question,

                context=context,
            )
        )


# ============================================================
# TTL RESPONSE CACHE
# ============================================================

class TTLResponseCache:

    """
    Small thread-safe in-memory TTL cache.

    Cache key:

        normalized question + top_k

    Cache value:

        complete RAG result

    IMPORTANT:

    This cache lives inside the application process.

    Therefore:

    - restarting Render clears it
    - multiple replicas would have separate caches

    That's acceptable for the current architecture.
    Redis would be the distributed replacement later.
    """

    def __init__(
        self,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        max_entries: int = CACHE_MAX_ENTRIES,
    ):

        self.ttl_seconds = ttl_seconds

        self.max_entries = max_entries

        self._cache = {}

        self._lock = threading.Lock()


    def _remove_expired(self):

        now = time.time()

        expired_keys = [

            key

            for key, item
            in self._cache.items()

            if (
                now
                - item["created_at"]
                > self.ttl_seconds
            )
        ]


        for key in expired_keys:

            self._cache.pop(
                key,
                None,
            )


    def get(
        self,
        key: str,
    ):

        with self._lock:

            self._remove_expired()


            item = self._cache.get(
                key
            )


            if item is None:

                return None


            # Return a copy so callers cannot mutate
            # the cached object.

            return deepcopy(
                item["value"]
            )


    def set(
        self,
        key: str,
        value,
    ):

        with self._lock:

            self._remove_expired()


            # Simple oldest-entry eviction.

            if (
                len(self._cache)
                >= self.max_entries
            ):

                oldest_key = min(

                    self._cache,

                    key=lambda item_key:
                        self._cache[
                            item_key
                        ][
                            "created_at"
                        ],
                )


                self._cache.pop(
                    oldest_key,
                    None,
                )


            self._cache[key] = {

                "created_at":
                    time.time(),

                "value":
                    deepcopy(value),
            }


    def clear(self):

        with self._lock:

            self._cache.clear()


    def size(self):

        with self._lock:

            self._remove_expired()

            return len(
                self._cache
            )


# ============================================================
# ENTERPRISE RAG
# ============================================================

class EnterpriseRAG:

    """
    Main Enterprise RAG application service.

    Features:

    - role-aware retrieval
    - version-aware caching
    - latency instrumentation
    - runtime metrics
    - Gemini/Groq generation routing
    """

    def __init__(
        self,
        retriever,
        generator,
        cache=None,
        metrics=None,
    ):

        self.retriever = (
            retriever
        )

        self.generator = (
            generator
        )

        self.cache = (

            cache

            if cache is not None

            else TTLResponseCache()
        )

        self.metrics = (

            metrics

            if metrics is not None

            else RuntimeMetrics()
        )


    # ========================================================
    # CACHE KEY
    # ========================================================

    @staticmethod
    def _build_cache_key(
        question: str,
        top_k: int,
        role: str,
    ) -> str:

        normalized_question = " ".join(

            question
            .lower()
            .strip()
            .split()
        )


        return (

            f"{KNOWLEDGE_BASE_VERSION}:"
            f"{role}:"
            f"{top_k}:"
            f"{normalized_question}"
        )


    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        role: str = DEFAULT_ROLE,
    ) -> Dict[str, Any]:

        total_started = (
            time.perf_counter()
        )


        # ====================================================
        # INPUT NORMALIZATION
        # ====================================================

        question = (
            question.strip()
        )


        role = (
            role
            .strip()
            .lower()
        )


        if not question:

            raise ValueError(
                "Question cannot be empty."
            )


        if role not in ALLOWED_ROLES:

            raise ValueError(
                f"Unsupported role: {role}"
            )


        effective_top_k = (

            top_k

            if top_k is not None

            else self.retriever.default_top_k
        )


        # ====================================================
        # CACHE KEY
        # ====================================================

        cache_key = (
            self._build_cache_key(

                question=
                    question,

                top_k=
                    effective_top_k,

                role=
                    role,
            )
        )


        # ====================================================
        # CACHE LOOKUP
        # ====================================================

        cache_started = (
            time.perf_counter()
        )


        cached_result = (
            self.cache.get(
                cache_key
            )
        )


        cache_lookup_ms = (

            time.perf_counter()
            - cache_started

        ) * 1000


        # ====================================================
        # CACHE HIT
        # ====================================================

        if cached_result is not None:

            total_ms = (

                time.perf_counter()
                - total_started

            ) * 1000


            cached_result[
                "cache_hit"
            ] = True


            cached_result[
                "timings"
            ] = {

                "cache_lookup_ms":
                    round(
                        cache_lookup_ms,
                        2,
                    ),

                "retrieval_ms":
                    0.0,

                "context_ms":
                    0.0,

                "generation_ms":
                    0.0,

                "total_ms":
                    round(
                        total_ms,
                        2,
                    ),
            }


            self.metrics.record_success(

                cache_hit=
                    True,

                provider=
                    cached_result.get(
                        "generation_provider"
                    ),

                latency_ms=
                    total_ms,
            )


            logger.info(
                "RAG_CACHE_HIT | "
                "role=%s | "
                "question=%r | "
                "top_k=%d | "
                "total_ms=%.2f",
                role,
                question,
                effective_top_k,
                total_ms,
            )


            return cached_result


        # ====================================================
        # CACHE MISS
        # ====================================================

        logger.info(
            "RAG_CACHE_MISS | "
            "role=%s | "
            "question=%r | "
            "top_k=%d",
            role,
            question,
            effective_top_k,
        )


        try:

            # =================================================
            # RETRIEVAL
            # =================================================

            retrieval_started = (
                time.perf_counter()
            )


            retrieved_chunks = (
                self.retriever.retrieve(

                    question=
                        question,

                    top_k=
                        effective_top_k,

                    role=
                        role,
                )
            )


            retrieval_ms = (

                time.perf_counter()
                - retrieval_started

            ) * 1000


            # =================================================
            # NO AUTHORIZED RESULTS
            # =================================================

            if not retrieved_chunks:

                total_ms = (

                    time.perf_counter()
                    - total_started

                ) * 1000


                result = {

                    "question":
                        question,

                    "role":
                        role,

                    "knowledge_base_version":
                        KNOWLEDGE_BASE_VERSION,

                    "answer":
                        (
                            "I don't know based on the "
                            "available policy documents."
                        ),

                    "generation_provider":
                        None,

                    "generation_model":
                        None,

                    "sources":
                        [],

                    "cache_hit":
                        False,

                    "timings": {

                        "cache_lookup_ms":
                            round(
                                cache_lookup_ms,
                                2,
                            ),

                        "retrieval_ms":
                            round(
                                retrieval_ms,
                                2,
                            ),

                        "context_ms":
                            0.0,

                        "generation_ms":
                            0.0,

                        "total_ms":
                            round(
                                total_ms,
                                2,
                            ),
                    },
                }


                self.cache.set(
                    cache_key,
                    result,
                )


                self.metrics.record_success(

                    cache_hit=
                        False,

                    provider=
                        None,

                    latency_ms=
                        total_ms,
                )


                return result


            # =================================================
            # CONTEXT
            # =================================================

            context_started = (
                time.perf_counter()
            )


            context = (
                ContextBuilder.build(
                    retrieved_chunks
                )
            )


            context_ms = (

                time.perf_counter()
                - context_started

            ) * 1000


            # =================================================
            # GENERATION
            # =================================================

            generation_started = (
                time.perf_counter()
            )


            generation_result = (
                self.generator.generate(

                    question=
                        question,

                    context=
                        context,
                )
            )


            generation_ms = (

                time.perf_counter()
                - generation_started

            ) * 1000


            # =================================================
            # SOURCES
            # =================================================

            sources = []


            for index, chunk in enumerate(
                retrieved_chunks,
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


                sources.append({

                    "number":
                        index,

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


            # =================================================
            # TOTAL
            # =================================================

            total_ms = (

                time.perf_counter()
                - total_started

            ) * 1000


            # =================================================
            # RESULT
            # =================================================

            result = {

                "question":
                    question,

                "role":
                    role,

                "knowledge_base_version":
                    KNOWLEDGE_BASE_VERSION,

                "answer":
                    generation_result[
                        "answer"
                    ],

                "generation_provider":
                    generation_result[
                        "provider"
                    ],

                "generation_model":
                    generation_result[
                        "model"
                    ],

                "sources":
                    sources,

                "cache_hit":
                    False,

                "timings": {

                    "cache_lookup_ms":
                        round(
                            cache_lookup_ms,
                            2,
                        ),

                    "retrieval_ms":
                        round(
                            retrieval_ms,
                            2,
                        ),

                    "context_ms":
                        round(
                            context_ms,
                            2,
                        ),

                    "generation_ms":
                        round(
                            generation_ms,
                            2,
                        ),

                    "total_ms":
                        round(
                            total_ms,
                            2,
                        ),
                },
            }


            # =================================================
            # CACHE
            # =================================================

            self.cache.set(

                cache_key,

                result,
            )


            # =================================================
            # METRICS
            # =================================================

            self.metrics.record_success(

                cache_hit=
                    False,

                provider=
                    result.get(
                        "generation_provider"
                    ),

                latency_ms=
                    total_ms,
            )


            # =================================================
            # OBSERVABILITY
            # =================================================

            logger.info(
                "RAG_TIMING | "
                "role=%s | "
                "kb_version=%s | "
                "provider=%s | "
                "model=%s | "
                "cache_hit=false | "
                "retrieval_ms=%.2f | "
                "context_ms=%.2f | "
                "generation_ms=%.2f | "
                "total_ms=%.2f",
                role,
                KNOWLEDGE_BASE_VERSION,
                result[
                    "generation_provider"
                ],
                result[
                    "generation_model"
                ],
                retrieval_ms,
                context_ms,
                generation_ms,
                total_ms,
            )


            return result


        except Exception:

            self.metrics.record_error()

            raise

model_circuit_breaker = (
    ModelCircuitBreaker()
)




# ============================================================
# RUNTIME METRICS
# ============================================================

class RuntimeMetrics:

    """
    Lightweight process-local metrics.

    No Prometheus/Redis/Datadog dependency yet.

    Tracks:

    - RAG requests
    - cache performance
    - provider usage
    - fallbacks
    - errors
    - latency
    """

    def __init__(self):

        self._lock = (
            threading.Lock()
        )

        self.started_at = (
            time.time()
        )

        self.total_requests = 0

        self.cache_hits = 0

        self.cache_misses = 0

        self.gemini_generations = 0

        self.groq_generations = 0

        self.fallbacks = 0

        self.errors = 0

        self.total_latency_ms = 0.0


    def record_success(
        self,
        cache_hit: bool,
        provider: Optional[str],
        latency_ms: float,
    ):

        with self._lock:

            self.total_requests += 1

            self.total_latency_ms += (
                latency_ms
            )


            if cache_hit:

                self.cache_hits += 1

            else:

                self.cache_misses += 1


            if provider == "gemini":

                self.gemini_generations += 1


            elif provider == "groq":

                self.groq_generations += 1

                # Groq is our cross-provider fallback.

                self.fallbacks += 1


    def record_error(self):

        with self._lock:

            self.total_requests += 1

            self.errors += 1


    def snapshot(
        self,
        cache_size: int,
    ) -> Dict[str, Any]:

        with self._lock:

            requests = (
                self.total_requests
            )


            cache_requests = (

                self.cache_hits
                + self.cache_misses
            )


            cache_hit_rate = (

                self.cache_hits
                / cache_requests

                if cache_requests

                else 0.0
            )


            average_latency = (

                self.total_latency_ms
                / requests

                if requests

                else 0.0
            )


            return {

                "uptime_seconds":
                    round(
                        time.time()
                        - self.started_at,
                        2,
                    ),

                "requests": {

                    "total":
                        requests,

                    "errors":
                        self.errors,
                },

                "cache": {

                    "hits":
                        self.cache_hits,

                    "misses":
                        self.cache_misses,

                    "hit_rate":
                        round(
                            cache_hit_rate,
                            4,
                        ),

                    "entries":
                        cache_size,

                    "ttl_seconds":
                        CACHE_TTL_SECONDS,
                },

                "generation": {

                    "gemini":
                        self.gemini_generations,

                    "groq":
                        self.groq_generations,

                    "provider_fallbacks":
                        self.fallbacks,
                },

                "latency": {

                    "average_ms":
                        round(
                            average_latency,
                            2,
                        )
                },

                "knowledge_base": {

                    "collection":
                        COLLECTION_NAME,

                    "version":
                        KNOWLEDGE_BASE_VERSION,
                },
            }


# ============================================================
# SHARED RUNTIME METRICS
# ============================================================

runtime_metrics = (
    RuntimeMetrics()
)




# ============================================================
# FACTORY
# ============================================================

def create_rag() -> EnterpriseRAG:

    """
    Create all application dependencies once.

    FastAPI calls this when the backend starts.
    """

    logger.info(
        "Initializing Enterprise Policy RAG..."
    )


    # ========================================================
    # GEMINI CLIENT
    # ========================================================

    gemini_client = genai.Client(

        api_key=
            GEMINI_API_KEY
    )


    # ========================================================
    # GROQ CLIENT
    # ========================================================

    groq_client = Groq(

        api_key=
            GROQ_API_KEY
    )


    # ========================================================
    # QDRANT CLIENT
    # ========================================================

    qdrant_client = QdrantClient(

        url=
            QDRANT_URL,

        api_key=
            QDRANT_API_KEY,

        timeout=
            60,
    )


    # ========================================================
    # EMBEDDING SERVICE
    # ========================================================

    embedding_service = (
        GeminiEmbeddingService(

            client=
                gemini_client,

            model=
                EMBEDDING_MODEL,

            retries=
                3,
        )
    )


    # ========================================================
    # VECTOR STORE
    # ========================================================

    vector_store = (
        QdrantVectorStore(

            client=
                qdrant_client,

            collection_name=
                COLLECTION_NAME,
        )
    )


    # ========================================================
    # RETRIEVER
    # ========================================================

    retriever = Retriever(

        embedding_service=
            embedding_service,

        vector_store=
            vector_store,

        default_top_k=
            TOP_K,
    )


    # ========================================================
    # GEMINI GENERATION SERVICE
    # ========================================================

    gemini_generator = (
        GeminiGenerationService(

            client=
                gemini_client,

            models=
                GEMINI_GENERATION_MODELS,

            circuit_breaker=
                model_circuit_breaker,
        )
    )

        # ========================================================
        # GROQ GENERATION SERVICE
        # ========================================================



    groq_generator = (
        GroqGenerationService(

            client=
                groq_client,

            models=
                GROQ_GENERATION_MODELS,

            circuit_breaker=
                model_circuit_breaker,
        )
    )

        # ========================================================
        # GENERATION ROUTER
        # ========================================================

    generation_router = (
        GenerationRouter(

            gemini_generator=
                gemini_generator,

            groq_generator=
                groq_generator,
        )
    )


    # ========================================================
    # COMPLETE RAG SERVICE
    # ========================================================

    rag = EnterpriseRAG(

        retriever=
            retriever,

        generator=
            generation_router,

        metrics=
            runtime_metrics,
    )


    logger.info(
        "Enterprise Policy RAG initialized."
    )


    return rag