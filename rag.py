# rag.py

import os

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

from google import genai

from qdrant_client import QdrantClient


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing."
    )

if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL is missing."
    )

if not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_API_KEY is missing."
    )


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-2"

GENERATION_MODEL = "gemini-3.8-flash"

COLLECTION_NAME = "acme_employee_policies"

TOP_K = 5


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
# EMBEDDING SERVICE
# ============================================================

class GeminiEmbeddingService:

    def __init__(
        self,
        client,
        model: str
    ):

        self.client = client

        self.model = model


    def embed_text(
        self,
        text: str
    ) -> List[float]:

        response = (
            self.client.models.embed_content(

                model=self.model,

                contents=text
            )
        )


        return list(
            response.embeddings[0].values
        )


# ============================================================
# QDRANT VECTOR STORE
# ============================================================

class QdrantVectorStore:

    def __init__(
        self,
        client,
        collection_name: str
    ):

        self.client = client

        self.collection_name = (
            collection_name
        )


    def search(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[RetrievedChunk]:

        response = (
            self.client.query_points(

                collection_name=
                    self.collection_name,

                query=query_vector,

                limit=top_k,

                with_payload=True
            )
        )


        results = []


        for point in response.points:

            payload = point.payload or {}


            results.append(

                RetrievedChunk(

                    chunk_id=payload.get(
                        "chunk_id",
                        str(point.id)
                    ),

                    text=payload.get(
                        "text",
                        ""
                    ),

                    score=float(
                        point.score
                    ),

                    source=payload.get(
                        "source",
                        ""
                    ),

                    heading_1=payload.get(
                        "heading_1"
                    ),

                    heading_2=payload.get(
                        "heading_2"
                    ),

                    heading_3=payload.get(
                        "heading_3"
                    )
                )
            )


        return results


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:

    def __init__(
        self,
        embedding_service,
        vector_store
    ):

        self.embedding_service = (
            embedding_service
        )

        self.vector_store = (
            vector_store
        )


    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K
    ) -> List[RetrievedChunk]:

        if not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )


        # Convert user question into
        # the same embedding space as
        # the document chunks.

        query_vector = (
            self.embedding_service
            .embed_text(question)
        )


        # Search existing vectors
        # stored permanently in Qdrant.

        return self.vector_store.search(

            query_vector=query_vector,

            top_k=top_k
        )


# ============================================================
# CONTEXT BUILDER
# ============================================================

class ContextBuilder:

    @staticmethod
    def build(
        chunks: List[RetrievedChunk]
    ) -> str:

        blocks = []


        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            headings = [

                chunk.heading_1,

                chunk.heading_2,

                chunk.heading_3
            ]


            heading_path = " > ".join(

                heading

                for heading in headings

                if heading
            )


            block = f"""
[SOURCE {index}]

Document:
{chunk.source}

Section:
{heading_path or "Unknown"}

Retrieval Score:
{chunk.score:.4f}

Content:
{chunk.text}
""".strip()


            blocks.append(block)


        return "\n\n---\n\n".join(
            blocks
        )


# ============================================================
# GENERATION SERVICE
# ============================================================

class GeminiGenerationService:

    def __init__(
        self,
        client,
        model: str
    ):

        self.client = client

        self.model = model


    def generate(
        self,
        question: str,
        context: str
    ) -> str:

        prompt = f"""
You are the employee-policy assistant for
Acme Global Services.

Answer employee questions using ONLY the
policy excerpts supplied below.

RULES:

1. Do not invent company policies.

2. Do not use general world knowledge as
   company policy.

3. Mention important limits, conditions,
   exceptions and approval requirements.

4. Cite factual statements using [Source N].

5. If the supplied context cannot answer
   the question, say:

   "I don't know based on the available
   policy documents."

6. If multiple policies apply, combine
   them carefully.


POLICY CONTEXT:

{context}


EMPLOYEE QUESTION:

{question}


ANSWER:
"""


        response = (
            self.client.models
            .generate_content(

                model=self.model,

                contents=prompt
            )
        )


        return response.text


# ============================================================
# ENTERPRISE RAG SERVICE
# ============================================================

class EnterpriseRAG:

    def __init__(
        self,
        retriever,
        generator
    ):

        self.retriever = retriever

        self.generator = generator


    def ask(
        self,
        question: str,
        top_k: int = TOP_K
    ) -> Dict[str, Any]:

        # -------------------------
        # RETRIEVAL
        # -------------------------

        retrieved = (
            self.retriever.retrieve(

                question=question,

                top_k=top_k
            )
        )


        if not retrieved:

            return {

                "question":
                    question,

                "answer":
                    "I don't know based on the available policy documents.",

                "sources":
                    []
            }


        # -------------------------
        # AUGMENTATION
        # -------------------------

        context = (
            ContextBuilder.build(
                retrieved
            )
        )


        # -------------------------
        # GENERATION
        # -------------------------

        answer = (
            self.generator.generate(

                question=question,

                context=context
            )
        )


        # -------------------------
        # SOURCE INFORMATION
        # -------------------------

        sources = []


        for index, chunk in enumerate(
            retrieved,
            start=1
        ):

            section = " > ".join(

                heading

                for heading in [

                    chunk.heading_1,
                    chunk.heading_2,
                    chunk.heading_3

                ]

                if heading
            )


            sources.append({

                "number":
                    index,

                "score":
                    round(
                        chunk.score,
                        4
                    ),

                "source":
                    chunk.source,

                "section":
                    section,

                "text":
                    chunk.text
            })


        return {

            "question":
                question,

            "answer":
                answer,

            "sources":
                sources
        }


# ============================================================
# FACTORY
# ============================================================

def create_rag() -> EnterpriseRAG:

    # Gemini connection

    gemini_client = genai.Client(

        api_key=
            GEMINI_API_KEY
    )


    # Qdrant connection

    qdrant_client = QdrantClient(

        url=QDRANT_URL,

        api_key=
            QDRANT_API_KEY,

        timeout=60
    )


    embedding_service = (
        GeminiEmbeddingService(

            client=
                gemini_client,

            model=
                EMBEDDING_MODEL
        )
    )


    vector_store = (
        QdrantVectorStore(

            client=
                qdrant_client,

            collection_name=
                COLLECTION_NAME
        )
    )


    retriever = Retriever(

        embedding_service=
            embedding_service,

        vector_store=
            vector_store
    )


    generator = (
        GeminiGenerationService(

            client=
                gemini_client,

            model=
                GENERATION_MODEL
        )
    )


    return EnterpriseRAG(

        retriever=
            retriever,

        generator=
            generator
    )