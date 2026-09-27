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

GROQ_FALLBACK_MODEL = (
    "openai/gpt-oss-120b"
)


# ------------------------------------------------------------
# QDRANT COLLECTION
# ------------------------------------------------------------

COLLECTION_NAME = (
    "acme_employee_policies"
)


# ------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------

TOP_K = 10


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
    Generates an embedding for the USER QUESTION.

    Document embeddings are already permanently stored
    inside Qdrant.
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

                        model=self.model,

                        contents=text,
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


                # --------------------------------------------
                # Retry only temporary service failures.
                # --------------------------------------------

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
                            ** (attempt - 1)
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
    ) -> List[RetrievedChunk]:

        logger.info(
            "Searching Qdrant | "
            "collection=%s | "
            "top_k=%d",
            self.collection_name,
            top_k,
        )


        response = (
            self.client.query_points(

                collection_name=
                    self.collection_name,

                query=
                    query_vector,

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
            "Qdrant returned %d chunks",
            len(results),
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
    ) -> List[RetrievedChunk]:

        question = (
            question.strip()
        )


        if not question:

            raise ValueError(
                "Question cannot be empty."
            )


        # ====================================================
        # STEP 1
        #
        # Generate query embedding.
        # ====================================================

        query_vector = (
            self.embedding_service
            .embed_text(
                question
            )
        )


        # ====================================================
        # STEP 2
        #
        # Search existing Qdrant vectors.
        # ====================================================

        return self.vector_store.search(

            query_vector=
                query_vector,

            top_k=
                top_k
                or self.default_top_k,
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
# GEMINI GENERATION SERVICE
# ============================================================

class GeminiGenerationService:

    """
    Gemini generation service.

    Attempts Gemini models sequentially.

    Example:

        Gemini model A
             |
            503
             |
           retry
             |
            503
             |
        Gemini model B
             |
           success

    IMPORTANT:

    This service returns:

        {
            "answer": "...",
            "provider": "gemini",
            "model": "..."
        }

    This is what fixes the missing provider/model metadata.
    """

    def __init__(
        self,
        client,
        models: List[str],
        retries_per_model: int = 2,
        initial_retry_delay: float = 1.5,
    ):

        self.client = client

        self.models = models

        self.retries_per_model = (
            retries_per_model
        )

        self.initial_retry_delay = (
            initial_retry_delay
        )


    @staticmethod
    def _is_retryable(
        error: Exception,
    ) -> bool:

        status_code = getattr(
            error,
            "status_code",
            None,
        )


        return status_code in {

            429,
            500,
            502,
            503,
            504,

        }


    def _generate_with_model(
        self,
        model: str,
        prompt: str,
    ) -> str:

        """
        Generate using Gemini.

        Keeping generate_content here means this remains
        compatible with the Gemini setup you already have
        working locally.
        """

        response = (
            self.client.models
            .generate_content(

                model=model,

                contents=prompt,
            )
        )


        answer = (
            response.text
        )


        if not answer:

            raise RuntimeError(
                f"Gemini model {model} "
                f"returned an empty response."
            )


        return answer


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


        # ====================================================
        # TRY EACH GEMINI MODEL
        # ====================================================

        for model in self.models:


            # =================================================
            # RETRIES FOR CURRENT MODEL
            # =================================================

            for attempt in range(
                1,
                self.retries_per_model + 1,
            ):

                try:

                    logger.info(
                        "Generation attempt | "
                        "provider=gemini | "
                        "model=%s | "
                        "attempt=%d/%d",
                        model,
                        attempt,
                        self.retries_per_model,
                    )


                    answer = (
                        self._generate_with_model(

                            model=model,

                            prompt=prompt,
                        )
                    )


                    logger.info(
                        "Generation successful | "
                        "provider=gemini | "
                        "model=%s",
                        model,
                    )


                    # ========================================
                    # IMPORTANT FIX
                    #
                    # Return answer + provider + actual model.
                    # ========================================

                    return {

                        "answer":
                            answer,

                        "provider":
                            "gemini",

                        "model":
                            model,
                    }


                except Exception as error:

                    last_error = error


                    status_code = getattr(
                        error,
                        "status_code",
                        None,
                    )


                    # ========================================
                    # NON-RETRYABLE ERROR
                    # ========================================

                    if not self._is_retryable(
                        error
                    ):

                        logger.warning(
                            "Gemini model failed with "
                            "non-retryable error | "
                            "model=%s | "
                            "status=%s | "
                            "error=%s",
                            model,
                            status_code,
                            error,
                        )


                        break


                    # ========================================
                    # RETRYABLE ERROR
                    # ========================================

                    logger.warning(
                        "Temporary Gemini failure | "
                        "model=%s | "
                        "status=%s | "
                        "attempt=%d/%d",
                        model,
                        status_code,
                        attempt,
                        self.retries_per_model,
                    )


                    if (
                        attempt
                        < self.retries_per_model
                    ):

                        delay = (

                            self.initial_retry_delay

                            * (

                                2

                                ** (
                                    attempt - 1
                                )
                            )
                        )


                        logger.info(
                            "Retrying %s "
                            "in %.1f seconds",
                            model,
                            delay,
                        )


                        time.sleep(
                            delay
                        )


            # =================================================
            # CURRENT MODEL FAILED
            # =================================================

            logger.warning(
                "Moving to next Gemini model | "
                "failed_model=%s",
                model,
            )


        # ====================================================
        # ALL GEMINI MODELS FAILED
        # ====================================================

        raise RuntimeError(
            "All configured Gemini "
            "generation models failed."
        ) from last_error


# ============================================================
# GROQ GENERATION SERVICE
# ============================================================

class GroqGenerationService:

    """
    Final cross-provider fallback.

    Used only if the complete Gemini chain fails.

    IMPORTANT:

    Returns the SAME dictionary structure as Gemini.
    """

    def __init__(
        self,
        client: Groq,
        model: str,
        retries: int = 2,
    ):

        self.client = client

        self.model = model

        self.retries = retries


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


        for attempt in range(
            1,
            self.retries + 1,
        ):

            try:

                logger.info(
                    "Generation attempt | "
                    "provider=groq | "
                    "model=%s | "
                    "attempt=%d/%d",
                    self.model,
                    attempt,
                    self.retries,
                )


                response = (
                    self.client
                    .chat
                    .completions
                    .create(

                        model=
                            self.model,

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
                        "Groq returned "
                        "an empty response."
                    )


                logger.info(
                    "Generation successful | "
                    "provider=groq | "
                    "model=%s",
                    self.model,
                )


                # ============================================
                # IMPORTANT FIX
                #
                # Same response contract as Gemini.
                # ============================================

                return {

                    "answer":
                        answer,

                    "provider":
                        "groq",

                    "model":
                        self.model,
                }


            except Exception as error:

                last_error = error


                logger.warning(
                    "Groq generation failed | "
                    "model=%s | "
                    "attempt=%d/%d | "
                    "error=%s",
                    self.model,
                    attempt,
                    self.retries,
                    error,
                )


                if attempt < self.retries:

                    delay = (
                        1.5
                        * (
                            2
                            ** (attempt - 1)
                        )
                    )


                    time.sleep(
                        delay
                    )


        raise RuntimeError(
            "Groq generation failed "
            "after all retries."
        ) from last_error


# ============================================================
# GENERATION ROUTER
# ============================================================

class GenerationRouter:

    """
    Cross-provider router.

    Flow:

        Gemini models
             |
        all fail
             |
             v
           Groq

    IMPORTANT:

    The entire result dictionary is returned.

    We do NOT return only result["answer"] because that would
    discard provider/model metadata.
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
        # PRIMARY: GEMINI
        # ====================================================

        try:

            result = (
                self.gemini_generator
                .generate(

                    question=question,

                    context=context,
                )
            )


            # IMPORTANT:
            #
            # Return entire result.
            #
            # NOT:
            #
            # return result["answer"]

            return result


        except Exception as gemini_error:

            logger.warning(
                "Gemini generation chain failed. "
                "Switching to Groq | "
                "error=%s",
                gemini_error,
            )


        # ====================================================
        # FALLBACK: GROQ
        # ====================================================

        try:

            result = (
                self.groq_generator
                .generate(

                    question=question,

                    context=context,
                )
            )


            return result


        except Exception as groq_error:

            logger.exception(
                "All generation providers failed."
            )


            raise RuntimeError(
                "Unable to generate an answer because "
                "all configured generation providers failed."
            ) from groq_error


# ============================================================
# ENTERPRISE RAG
# ============================================================

class EnterpriseRAG:

    """
    Main RAG application service.

    Used by:

        FastAPI
        CLI
        Streamlit (if desired)

    Flow:

        question
           |
        retrieve
           |
        context
           |
        generation router
           |
        answer + metadata + sources
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: GenerationRouter,
    ):

        self.retriever = (
            retriever
        )

        self.generator = (
            generator
        )


    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:

        question = (
            question.strip()
        )


        if not question:

            raise ValueError(
                "Question cannot be empty."
            )


        logger.info(
            "RAG request started | "
            "question=%s",
            question,
        )


        # ====================================================
        # 1. RETRIEVAL
        # ====================================================

        retrieved_chunks = (
            self.retriever.retrieve(

                question=question,

                top_k=top_k,
            )
        )


        if not retrieved_chunks:

            return {

                "question":
                    question,

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
            }


        # ====================================================
        # 2. AUGMENTATION
        # ====================================================

        context = (
            ContextBuilder.build(
                retrieved_chunks
            )
        )


        # ====================================================
        # 3. GENERATION
        # ====================================================

        generation_result = (
            self.generator.generate(

                question=question,

                context=context,
            )
        )


        # ====================================================
        # 4. BUILD SOURCE METADATA
        # ====================================================

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


        # ====================================================
        # 5. FINAL RESULT
        # ====================================================

        result = {

            "question":
                question,

            "answer":
                generation_result[
                    "answer"
                ],

            # ================================================
            # IMPORTANT FIX
            # ================================================

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
        }


        logger.info(
            "RAG request completed | "
            "provider=%s | "
            "model=%s",
            result[
                "generation_provider"
            ],
            result[
                "generation_model"
            ],
        )


        return result


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

            retries_per_model=
                2,

            initial_retry_delay=
                1.5,
        )
    )


    # ========================================================
    # GROQ GENERATION SERVICE
    # ========================================================

    groq_generator = (
        GroqGenerationService(

            client=
                groq_client,

            model=
                GROQ_FALLBACK_MODEL,

            retries=
                2,
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
    )


    logger.info(
        "Enterprise Policy RAG initialized."
    )


    return rag