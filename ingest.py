"""
ingest.py

Enterprise Policy RAG - Production-Style Ingestion Pipeline

Source:
    documents/enterprise_policy_handbook.md

Destination:
    Qdrant collection:
    acme_employee_policies_v2

Pipeline:

    Markdown handbook
        |
        v
    Parse categories / policies
        |
        v
    Validate policy structure
        |
        v
    Policy-aware chunking
        |
        v
    Validate chunks
        |
        v
    Build retrieval-optimized embedding text
        |
        v
    Gemini Embedding 2
        |
        | batched independent Content objects
        v
    Validate embeddings
        |
        v
    Create Qdrant V2 collection
        |
        v
    Upload vectors + metadata
        |
        v
    Validate collection
        |
        v
    Retrieval smoke tests

IMPORTANT:

This script ONLY recreates:

    acme_employee_policies_v2

It does NOT modify:

    acme_employee_policies

Run:

    python ingest.py
"""

# ============================================================
# IMPORTS
# ============================================================

import hashlib
import logging
import os
import re
import time
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
)

from dotenv import load_dotenv

from google import genai
from google.genai import types

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "enterprise-rag-ingestion"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL"
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)


REQUIRED_ENVIRONMENT_VARIABLES = {

    "GEMINI_API_KEY":
        GEMINI_API_KEY,

    "QDRANT_URL":
        QDRANT_URL,

    "QDRANT_API_KEY":
        QDRANT_API_KEY,
}


for name, value in (
    REQUIRED_ENVIRONMENT_VARIABLES.items()
):

    if not value:

        raise RuntimeError(
            f"Missing environment variable: {name}"
        )


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


DOCUMENT_PATH = (

    BASE_DIR
    / "documents"
    / "enterprise_policy_handbook.md"
)


SOURCE_NAME = (
    "enterprise_policy_handbook.md"
)


# ============================================================
# DOCUMENT CONFIGURATION
# ============================================================

DOCUMENT_VERSION = "2.0"


# ============================================================
# QDRANT CONFIGURATION
# ============================================================

# IMPORTANT:
#
# This is intentionally a NEW collection.
#
# Your existing collection remains untouched.

COLLECTION_NAME = (
    "acme_employee_policies_v2"
)


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL = (
    "gemini-embedding-2"
)


# Your existing Gemini Embedding 2 vectors are 3072-dimensional.

EXPECTED_EMBEDDING_DIMENSION = 3072


# ============================================================
# BATCH CONFIGURATION
# ============================================================

# 20 chunks are sent in ONE API request.
#
# Each chunk is represented as an independent types.Content
# object so Gemini returns one embedding per chunk.

EMBEDDING_BATCH_SIZE = 20


# Number of vectors uploaded to Qdrant per request.

QDRANT_BATCH_SIZE = 50


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

# Markdown subsections are our preferred natural boundaries.
#
# This value is only used when a subsection becomes unusually
# large.

MAX_CHUNK_CHARS = 3000


# Overlap applies ONLY when splitting a large subsection.
#
# It never crosses policy boundaries.

CHUNK_OVERLAP_CHARS = 250


# ============================================================
# RETRY CONFIGURATION
# ============================================================

EMBEDDING_MAX_RETRIES = 5

MAX_RETRY_DELAY_SECONDS = 30


EMBEDDING_INPUTS_PER_MINUTE = 80
RATE_LIMIT_WINDOW_SECONDS = 60

# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class PolicySection:

    category: str

    policy: str

    policy_id: str

    content: str


@dataclass
class PolicyChunk:

    chunk_id: str

    category: str

    policy: str

    policy_id: str

    subsection: Optional[str]

    text: str

    embedding_text: str

    chunk_index: int

    source: str

    document_version: str


# ============================================================
# CLIENTS
# ============================================================

gemini_client = genai.Client(

    api_key=
        GEMINI_API_KEY
)


qdrant_client = QdrantClient(

    url=
        QDRANT_URL,

    api_key=
        QDRANT_API_KEY,

    timeout=
        60,
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str,
) -> str:

    """
    Normalize whitespace while preserving paragraph structure.
    """

    text = text.replace(
        "\xa0",
        " ",
    )


    # Normalize spaces/tabs.

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )


    # Avoid excessive blank lines.

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )


    return text.strip()


# ============================================================
# STABLE CHUNK ID
# ============================================================

def create_chunk_id(
    policy_id: str,
    subsection: Optional[str],
    chunk_index: int,
    text: str,
) -> str:

    """
    Generate a deterministic logical chunk ID.

    If the same policy content is ingested again with the same
    document version, it receives the same chunk ID.
    """

    raw = (
        f"{DOCUMENT_VERSION}|"
        f"{policy_id}|"
        f"{subsection or ''}|"
        f"{chunk_index}|"
        f"{text}"
    )


    digest = hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()


    return digest[:32]


# ============================================================
# MARKDOWN PARSER
# ============================================================

def parse_markdown_document(
    file_path: Path,
) -> List[PolicySection]:

    """
    Parse the structured Markdown policy handbook.

    Expected hierarchy:

        # Handbook

        ## Benefits

        ### Loans & Advances

        **Policy ID:** BEN-006

        #### Purpose
        ...

        #### Scope
        ...

        #### Policy Rules
        ...

    Interpretation:

        ##     category
        ###    policy
        ####   policy subsection

    Only ### sections containing a valid Policy ID are treated
    as primary policies.

    This intentionally prevents front matter, navigation,
    glossary material and other non-policy content from
    entering the primary vector index.
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"Markdown document not found: "
            f"{file_path}"
        )


    logger.info(
        "Reading Markdown document: %s",
        file_path,
    )


    markdown = file_path.read_text(
        encoding="utf-8"
    )


    lines = markdown.splitlines()


    policies: List[PolicySection] = []


    current_category = None

    current_policy = None

    policy_buffer: List[str] = []


    # ========================================================
    # FLUSH CURRENT POLICY
    # ========================================================

    def flush_policy():

        nonlocal policy_buffer


        if (
            not current_category
            or not current_policy
            or not policy_buffer
        ):

            policy_buffer = []

            return


        content = normalize_text(

            "\n".join(
                policy_buffer
            )
        )


        # ----------------------------------------------------
        # Extract stable Policy ID.
        #
        # Example:
        #
        # **Policy ID:** BEN-006
        # ----------------------------------------------------

        policy_id_match = re.search(

            r"\*\*Policy ID:\*\*\s*"
            r"([A-Z][A-Z0-9-]+)",

            content,
        )


        # No Policy ID means this isn't a primary policy.

        if not policy_id_match:

            logger.debug(
                "Skipping non-policy section | "
                "category=%s | policy=%s",
                current_category,
                current_policy,
            )


            policy_buffer = []

            return


        policy_id = (
            policy_id_match
            .group(1)
            .strip()
        )


        policies.append(

            PolicySection(

                category=
                    current_category,

                policy=
                    current_policy,

                policy_id=
                    policy_id,

                content=
                    content,
            )
        )


        policy_buffer = []


    # ========================================================
    # PARSE DOCUMENT
    # ========================================================

    for raw_line in lines:

        line = raw_line.rstrip()


        # ----------------------------------------------------
        # CATEGORY
        #
        # Exactly ## heading.
        #
        # Negative lookahead prevents matching ###.
        # ----------------------------------------------------

        if re.match(
            r"^##(?!#)\s+",
            line,
        ):

            flush_policy()


            current_category = re.sub(

                r"^##\s+",

                "",

                line,
            ).strip()


            current_policy = None

            policy_buffer = []


            continue


        # ----------------------------------------------------
        # POLICY
        #
        # Exactly ### heading.
        # ----------------------------------------------------

        if re.match(
            r"^###(?!#)\s+",
            line,
        ):

            flush_policy()


            current_policy = re.sub(

                r"^###\s+",

                "",

                line,
            ).strip()


            policy_buffer = []


            continue


        # ----------------------------------------------------
        # POLICY CONTENT
        # ----------------------------------------------------

        if current_policy:

            policy_buffer.append(
                line
            )


    # Flush final policy.

    flush_policy()


    logger.info(
        "Parsed %d primary policies.",
        len(policies),
    )


    return policies


# ============================================================
# POLICY VALIDATION
# ============================================================

def validate_policies(
    policies: List[PolicySection],
) -> None:

    """
    Validate document structure BEFORE using embedding quota.
    """

    if not policies:

        raise RuntimeError(
            "No policies were parsed."
        )


    # ========================================================
    # DUPLICATE POLICY IDS
    # ========================================================

    policy_ids = [

        policy.policy_id

        for policy
        in policies
    ]


    duplicate_policy_ids = {

        policy_id

        for policy_id
        in policy_ids

        if policy_ids.count(
            policy_id
        ) > 1
    }


    if duplicate_policy_ids:

        raise RuntimeError(

            "Duplicate policy IDs detected: "

            + ", ".join(
                sorted(
                    duplicate_policy_ids
                )
            )
        )


    # ========================================================
    # REQUIRED POLICIES
    #
    # These include the policies from our evaluation suite,
    # especially the ones the old collection failed.
    # ========================================================

    required_policy_names = {

        "Loans & Advances",

        "Separation",

        "International Travel",

        "Health & Insurance",

        "Certification & Professional Membership",

        "Working Hours",

        "Leave",

        "Flexible Working",

        "Social Media",

        "Seminars & Conferences",
    }


    parsed_policy_names = {

        policy.policy

        for policy
        in policies
    }


    missing_policies = (

        required_policy_names
        - parsed_policy_names
    )


    if missing_policies:

        raise RuntimeError(

            "Critical policies missing from corpus: "

            + ", ".join(
                sorted(
                    missing_policies
                )
            )
        )


    logger.info(
        "Policy validation passed."
    )


# ============================================================
# SUBSECTION PARSER
# ============================================================

def split_policy_into_subsections(
    policy: PolicySection,
) -> List[Dict[str, Optional[str]]]:

    """
    Split an individual policy using #### headings.

    The metadata appearing before the first #### heading becomes
    a small policy-level metadata subsection.

    Example:

        Policy metadata
        ↓
        Purpose
        ↓
        Scope
        ↓
        Policy Rules
        ↓
        Eligibility
        ↓
        Common Questions
    """

    lines = (
        policy.content
        .splitlines()
    )


    subsections = []


    current_heading: Optional[str] = None

    current_buffer: List[str] = []


    # ========================================================
    # FLUSH SUBSECTION
    # ========================================================

    def flush():

        nonlocal current_buffer


        text = normalize_text(

            "\n".join(
                current_buffer
            )
        )


        if text:

            subsections.append({

                "heading":
                    current_heading,

                "text":
                    text,
            })


        current_buffer = []


    # ========================================================
    # PARSE SUBSECTIONS
    # ========================================================

    for line in lines:

        if re.match(
            r"^####\s+",
            line,
        ):

            flush()


            current_heading = re.sub(

                r"^####\s+",

                "",

                line,
            ).strip()


            continue


        current_buffer.append(
            line
        )


    flush()


    return subsections


# ============================================================
# LARGE SUBSECTION SPLITTER
# ============================================================

def split_large_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> List[str]:

    """
    Split unusually large subsections.

    Normal policies should already split cleanly by Markdown
    subsection.

    This function exists only as a fallback when an individual
    subsection exceeds MAX_CHUNK_CHARS.

    Overlap never crosses a policy/subsection boundary.
    """

    text = normalize_text(
        text
    )


    if len(text) <= max_chars:

        return [text]


    paragraphs = [

        normalize_text(
            paragraph
        )

        for paragraph
        in re.split(
            r"\n\s*\n",
            text,
        )

        if normalize_text(
            paragraph
        )
    ]


    chunks = []

    buffer = []

    buffer_length = 0


    for paragraph in paragraphs:

        projected_length = (

            buffer_length

            + len(paragraph)

            + 2
        )


        if (
            projected_length
            > max_chars
            and buffer
        ):

            chunk_text = normalize_text(

                "\n\n".join(
                    buffer
                )
            )


            chunks.append(
                chunk_text
            )


            # ------------------------------------------------
            # Controlled overlap.
            #
            # This overlap remains inside the same subsection.
            # ------------------------------------------------

            overlap = (

                chunk_text[
                    -overlap_chars:
                ]

                if overlap_chars > 0

                else ""
            )


            buffer = (

                [overlap]

                if overlap

                else []
            )


            buffer_length = len(
                overlap
            )


        buffer.append(
            paragraph
        )


        buffer_length += (
            len(paragraph)
            + 2
        )


    if buffer:

        final_chunk = normalize_text(

            "\n\n".join(
                buffer
            )
        )


        if final_chunk:

            chunks.append(
                final_chunk
            )


    return chunks


# ============================================================
# EMBEDDING REPRESENTATION
# ============================================================

def build_embedding_text(
    category: str,
    policy: str,
    policy_id: str,
    subsection: Optional[str],
    text: str,
) -> str:

    """
    Build the text actually sent to the embedding model.

    This is intentionally richer than the clean text stored for
    LLM generation.

    Example:

        Category: HR Processes

        Policy: Separation

        Policy ID: HRP-002

        Section: Policy Rules

        Employees who resign...

    This gives semantic retrieval strong signals about what
    policy the content belongs to.
    """

    parts = [

        f"Category: {category}",

        f"Policy: {policy}",

        f"Policy ID: {policy_id}",
    ]


    if subsection:

        parts.append(
            f"Section: {subsection}"
        )


    parts.append(
        text
    )


    return "\n\n".join(
        parts
    )


# ============================================================
# POLICY-AWARE CHUNKER
# ============================================================

def create_policy_chunks(
    policies: List[PolicySection],
) -> List[PolicyChunk]:

    """
    Build clean policy-aware chunks.

    Guarantees:

    - chunks do not cross policy boundaries
    - chunks do not blindly overlap between policies
    - category/policy metadata is available to embeddings
    - structured metadata is preserved for Qdrant
    """

    chunks: List[PolicyChunk] = []


    for policy in policies:

        subsections = (
            split_policy_into_subsections(
                policy
            )
        )


        policy_chunk_index = 0


        for subsection in subsections:

            subsection_heading = (
                subsection[
                    "heading"
                ]
            )


            subsection_text = (
                subsection[
                    "text"
                ]
                or ""
            )


            # Ignore meaningless fragments.

            if len(
                subsection_text.strip()
            ) < 20:

                continue


            pieces = split_large_text(

                text=
                    subsection_text,

                max_chars=
                    MAX_CHUNK_CHARS,

                overlap_chars=
                    CHUNK_OVERLAP_CHARS,
            )


            for piece in pieces:

                chunk_id = (
                    create_chunk_id(

                        policy_id=
                            policy.policy_id,

                        subsection=
                            subsection_heading,

                        chunk_index=
                            policy_chunk_index,

                        text=
                            piece,
                    )
                )


                embedding_text = (
                    build_embedding_text(

                        category=
                            policy.category,

                        policy=
                            policy.policy,

                        policy_id=
                            policy.policy_id,

                        subsection=
                            subsection_heading,

                        text=
                            piece,
                    )
                )


                chunks.append(

                    PolicyChunk(

                        chunk_id=
                            chunk_id,

                        category=
                            policy.category,

                        policy=
                            policy.policy,

                        policy_id=
                            policy.policy_id,

                        subsection=
                            subsection_heading,

                        text=
                            piece,

                        embedding_text=
                            embedding_text,

                        chunk_index=
                            policy_chunk_index,

                        source=
                            SOURCE_NAME,

                        document_version=
                            DOCUMENT_VERSION,
                    )
                )


                policy_chunk_index += 1


    logger.info(
        "Created %d policy-aware chunks.",
        len(chunks),
    )


    return chunks


# ============================================================
# CHUNK VALIDATION
# ============================================================

def validate_chunks(
    chunks: List[PolicyChunk],
) -> None:

    """
    Validate chunk quality BEFORE spending embedding quota.
    """

    if not chunks:

        raise RuntimeError(
            "No chunks were generated."
        )


    # ========================================================
    # UNIQUE IDS
    # ========================================================

    chunk_ids = [

        chunk.chunk_id

        for chunk
        in chunks
    ]


    if len(chunk_ids) != len(
        set(chunk_ids)
    ):

        raise RuntimeError(
            "Duplicate chunk IDs detected."
        )


    # ========================================================
    # REQUIRED METADATA
    # ========================================================

    for chunk in chunks:

        if not chunk.text.strip():

            raise RuntimeError(
                "Empty chunk detected: "
                f"{chunk.chunk_id}"
            )


        if not chunk.category.strip():

            raise RuntimeError(
                "Chunk missing category: "
                f"{chunk.chunk_id}"
            )


        if not chunk.policy.strip():

            raise RuntimeError(
                "Chunk missing policy: "
                f"{chunk.chunk_id}"
            )


        if not chunk.policy_id.strip():

            raise RuntimeError(
                "Chunk missing policy ID: "
                f"{chunk.chunk_id}"
            )


        if not chunk.embedding_text.strip():

            raise RuntimeError(
                "Chunk missing embedding text: "
                f"{chunk.chunk_id}"
            )


    # ========================================================
    # NAVIGATION / TOC DETECTION
    #
    # The previous collection was polluted by chunks that
    # contained many unrelated policy titles.
    #
    # Detect that BEFORE embedding.
    # ========================================================

    known_policy_names = {

        chunk.policy

        for chunk
        in chunks
    }


    suspicious_chunks = []


    for chunk in chunks:

        unrelated_policy_mentions = 0


        for policy_name in (
            known_policy_names
        ):

            if (
                policy_name
                != chunk.policy
                and policy_name
                in chunk.text
            ):

                unrelated_policy_mentions += 1


        # A real policy may reference one or two related
        # policies.
        #
        # Five unrelated policy titles strongly suggests
        # navigation / TOC contamination.

        if (
            unrelated_policy_mentions
            >= 5
        ):

            suspicious_chunks.append(
                chunk
            )


    if suspicious_chunks:

        examples = [

            (
                f"{chunk.policy_id} "
                f"{chunk.policy}"
            )

            for chunk
            in suspicious_chunks[:5]
        ]


        raise RuntimeError(

            "Navigation-like chunks detected: "

            + ", ".join(
                examples
            )
        )


    logger.info(
        "Chunk validation passed."
    )


# ============================================================
# GEMINI EMBEDDING SERVICE
# ============================================================

class GeminiEmbeddingService:

    """
    Gemini Embedding 2 service.

    IMPORTANT:

    Each chunk is represented as an independent types.Content
    object.

    Example:

        Content(chunk 1)
        Content(chunk 2)
        Content(chunk 3)

              ↓

        ONE API REQUEST

              ↓

        embedding 1
        embedding 2
        embedding 3

    This gives us efficient batching WITHOUT collapsing multiple
    chunks into a single aggregated embedding.
    """

    def __init__(
        self,
        client,
        model: str,
        batch_size: int,
        retries: int,
    ):

        self.client = client

        self.model = model

        self.batch_size = (
            batch_size
        )

        self.retries = retries


    # ========================================================
    # EMBED ONE BATCH
    # ========================================================

    def _embed_batch(
        self,
        texts: List[str],
    ):

        if not texts:

            raise ValueError(
                "Cannot embed an empty batch."
            )


        # ----------------------------------------------------
        # CRITICAL:
        #
        # Every chunk becomes its OWN Content object.
        # ----------------------------------------------------

        contents = [

            types.Content(

                parts=[

                    types.Part.from_text(
                        text=text
                    )

                ]
            )

            for text
            in texts
        ]


        last_error = None


        # ====================================================
        # RETRY LOOP
        # ====================================================

        for attempt in range(
            1,
            self.retries + 1,
        ):

            try:

                response = (
                    self.client.models
                    .embed_content(

                        model=
                            self.model,

                        contents=
                            contents,
                    )
                )


                if not response.embeddings:

                    raise RuntimeError(
                        "Gemini returned no embeddings."
                    )


                actual_count = len(
                    response.embeddings
                )


                expected_count = len(
                    texts
                )


                # --------------------------------------------
                # Critical one-to-one validation.
                # --------------------------------------------

                if (
                    actual_count
                    != expected_count
                ):

                    raise RuntimeError(

                        "Gemini embedding count mismatch. "
                        f"Sent {expected_count} independent "
                        f"Content objects but received "
                        f"{actual_count} embeddings."
                    )


                return response


            except RuntimeError:

                # Programming/data-shape errors should fail
                # immediately rather than wasting quota.

                raise


            except Exception as error:

                last_error = error


                status_code = getattr(
                    error,
                    "status_code",
                    None,
                )


                # --------------------------------------------
                # Retry ONLY transient API failures.
                # --------------------------------------------

                if status_code not in {

                    429,
                    500,
                    502,
                    503,
                    504,

                }:

                    raise


                if attempt >= self.retries:

                    break


                delay = min(

                    2 ** attempt,

                    MAX_RETRY_DELAY_SECONDS,
                )


                logger.warning(
                    "Embedding API temporary failure | "
                    "status=%s | "
                    "attempt=%d/%d | "
                    "retry_in=%ds",
                    status_code,
                    attempt,
                    self.retries,
                    delay,
                )


                time.sleep(
                    delay
                )


        raise RuntimeError(
            "Embedding batch failed after retries."
        ) from last_error


    # ========================================================
    # EMBED ALL DOCUMENT CHUNKS
    # ========================================================

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        if not texts:

            raise ValueError(
                "No document chunks supplied for embedding."
            )


        embeddings: List[
            List[float]
        ] = []


        total = len(
            texts
        )


        total_batches = (

            total
            + self.batch_size
            - 1

        ) // self.batch_size


        logger.info(
            "Generating embeddings | "
            "chunks=%d | "
            "batch_size=%d | "
            "estimated_requests=%d",
            total,
            self.batch_size,
            total_batches,
        )


        for batch_number, start in enumerate(

            range(
                0,
                total,
                self.batch_size,
            ),

            start=1,
        ):

            batch = texts[

                start:
                start
                + self.batch_size
            ]


            end = (
                start
                + len(batch)
            )


            logger.info(
                "Embedding batch %d/%d | "
                "chunks=%d-%d",
                batch_number,
                total_batches,
                start + 1,
                end,
            )


            response = (
                self._embed_batch(
                    batch
                )
            )


            batch_embeddings = [

                list(
                    embedding.values
                )

                for embedding
                in response.embeddings
            ]


            # Secondary validation.

            if (
                len(batch_embeddings)
                != len(batch)
            ):

                raise RuntimeError(

                    "Embedding count mismatch after "
                    "response conversion. "
                    f"Expected {len(batch)}, "
                    f"received "
                    f"{len(batch_embeddings)}."
                )


            embeddings.extend(
                batch_embeddings
            )

# ============================================================
# CLIENT-SIDE QUOTA THROTTLING
# ============================================================

# The Gemini embedding quota is consumed by the number of
# embedded inputs, not merely by the number of HTTP calls.
#
# Example:
#
#     batch size = 20
#
#     5 HTTP calls
#       =
#     100 embedding inputs
#
# Therefore batching improves network efficiency but does not
# bypass the embedding-input rate limit.
#
# We intentionally pause before entering the next quota window.

            if (
                len(embeddings) < total
                and len(embeddings) % EMBEDDING_INPUTS_PER_MINUTE == 0
            ):

                logger.info(
                    "Embedding quota window reached | "
                    "processed=%d/%d | "
                    "sleeping=%ds",
                    len(embeddings),
                    total,
                    RATE_LIMIT_WINDOW_SECONDS,
                )

                time.sleep(
                    RATE_LIMIT_WINDOW_SECONDS
                )

            logger.info(
                "Embedding progress | "
                "%d/%d chunks complete",
                len(embeddings),
                total,
            )


        return embeddings


    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_query(
        self,
        text: str,
    ) -> List[float]:

        """
        Used by post-ingestion retrieval smoke tests.
        """

        if not text.strip():

            raise ValueError(
                "Query cannot be empty."
            )


        response = (
            self._embed_batch(
                [text]
            )
        )


        return list(
            response
            .embeddings[0]
            .values
        )


# ============================================================
# EMBEDDING VALIDATION
# ============================================================

def validate_embeddings(
    embeddings: List[List[float]],
    expected_count: int,
) -> int:

    """
    Validate embeddings BEFORE creating the Qdrant collection.
    """

    if not embeddings:

        raise RuntimeError(
            "No embeddings were generated."
        )


    # ========================================================
    # COUNT
    # ========================================================

    actual_count = len(
        embeddings
    )


    if (
        actual_count
        != expected_count
    ):

        raise RuntimeError(

            "Embedding count mismatch. "
            f"Expected {expected_count}, "
            f"received {actual_count}."
        )


    # ========================================================
    # DIMENSIONS
    # ========================================================

    dimensions = {

        len(vector)

        for vector
        in embeddings
    }


    if len(dimensions) != 1:

        raise RuntimeError(

            "Embedding dimensions are inconsistent: "

            + str(
                sorted(dimensions)
            )
        )


    vector_size = next(
        iter(dimensions)
    )


    if (
        EXPECTED_EMBEDDING_DIMENSION
        and vector_size
        != EXPECTED_EMBEDDING_DIMENSION
    ):

        raise RuntimeError(

            "Unexpected embedding dimension. "
            f"Expected "
            f"{EXPECTED_EMBEDDING_DIMENSION}, "
            f"received {vector_size}."
        )


    logger.info(
        "Embedding validation passed | "
        "count=%d | "
        "dimension=%d",
        actual_count,
        vector_size,
    )


    return vector_size


# ============================================================
# QDRANT COLLECTION MANAGEMENT
# ============================================================

def recreate_v2_collection(
    vector_size: int,
) -> None:

    """
    Recreate ONLY acme_employee_policies_v2.

    The original collection is never modified by this function.
    """

    logger.info(
        "Checking Qdrant collections."
    )


    collections_response = (
        qdrant_client
        .get_collections()
    )


    existing_collections = {

        collection.name

        for collection
        in collections_response.collections
    }


    if (
        COLLECTION_NAME
        in existing_collections
    ):

        logger.warning(
            "Deleting existing V2 collection | "
            "collection=%s",
            COLLECTION_NAME,
        )


        qdrant_client.delete_collection(

            collection_name=
                COLLECTION_NAME
        )


    logger.info(
        "Creating Qdrant collection | "
        "collection=%s | "
        "dimension=%d | "
        "distance=cosine",
        COLLECTION_NAME,
        vector_size,
    )


    qdrant_client.create_collection(

        collection_name=
            COLLECTION_NAME,

        vectors_config=
            VectorParams(

                size=
                    vector_size,

                distance=
                    Distance.COSINE,
            ),
    )


# ============================================================
# QDRANT UPLOAD
# ============================================================

def upload_to_qdrant(
    chunks: List[PolicyChunk],
    embeddings: List[List[float]],
) -> None:

    """
    Convert chunks + embeddings into Qdrant points and upload
    them in batches.
    """

    if len(chunks) != len(
        embeddings
    ):

        raise RuntimeError(

            "Chunk / embedding count mismatch. "
            f"Chunks={len(chunks)}, "
            f"embeddings={len(embeddings)}"
        )


    points = []


    for chunk, vector in zip(
        chunks,
        embeddings,
    ):

        # ----------------------------------------------------
        # Qdrant requires integer or UUID point IDs.
        #
        # Derive deterministic UUID from our logical chunk ID.
        # ----------------------------------------------------

        point_id = str(

            uuid.uuid5(

                uuid.NAMESPACE_URL,

                chunk.chunk_id,
            )
        )


        payload = {

            # =================================================
            # IDENTIFIERS
            # =================================================

            "chunk_id":
                chunk.chunk_id,

            "policy_id":
                chunk.policy_id,


            # =================================================
            # DOCUMENT METADATA
            # =================================================

            "source":
                chunk.source,

            "document_version":
                chunk.document_version,

            "content_type":
                "policy",


            # =================================================
            # POLICY METADATA
            # =================================================

            "category":
                chunk.category,

            "policy":
                chunk.policy,

            "subsection":
                chunk.subsection,


            # =================================================
            # COMPATIBILITY WITH EXISTING rag.py
            #
            # Your current retriever reads:
            #
            # heading_1
            # heading_2
            # heading_3
            # =================================================

            "heading_1":
                chunk.category,

            "heading_2":
                chunk.policy,

            "heading_3":
                chunk.subsection,


            # =================================================
            # CLEAN RETRIEVED CONTENT
            #
            # This is the text supplied to the generation model.
            #
            # embedding_text itself is NOT stored as the primary
            # LLM context because it contains retrieval metadata.
            # =================================================

            "text":
                chunk.text,


            # =================================================
            # CHUNK METADATA
            # =================================================

            "chunk_index":
                chunk.chunk_index,
        }


        points.append(

            PointStruct(

                id=
                    point_id,

                vector=
                    vector,

                payload=
                    payload,
            )
        )


    total = len(
        points
    )


    total_batches = (

        total
        + QDRANT_BATCH_SIZE
        - 1

    ) // QDRANT_BATCH_SIZE


    logger.info(
        "Uploading vectors to Qdrant | "
        "points=%d | "
        "batch_size=%d | "
        "batches=%d",
        total,
        QDRANT_BATCH_SIZE,
        total_batches,
    )


    for batch_number, start in enumerate(

        range(
            0,
            total,
            QDRANT_BATCH_SIZE,
        ),

        start=1,
    ):

        batch = points[

            start:
            start
            + QDRANT_BATCH_SIZE
        ]


        qdrant_client.upsert(

            collection_name=
                COLLECTION_NAME,

            points=
                batch,

            wait=
                True,
        )


        logger.info(
            "Qdrant upload | "
            "batch=%d/%d | "
            "points=%d-%d",
            batch_number,
            total_batches,
            start + 1,
            min(
                start + len(batch),
                total,
            ),
        )


# ============================================================
# QDRANT VALIDATION
# ============================================================

def validate_qdrant_collection(
    expected_count: int,
) -> None:

    """
    Validate the collection after upload.
    """

    collection_info = (
        qdrant_client
        .get_collection(

            collection_name=
                COLLECTION_NAME
        )
    )


    points_count = getattr(
        collection_info,
        "points_count",
        None,
    )


    logger.info(
        "Qdrant validation | "
        "expected_points=%d | "
        "actual_points=%s",
        expected_count,
        points_count,
    )


    if (
        points_count is not None
        and points_count
        != expected_count
    ):

        raise RuntimeError(

            "Qdrant point count mismatch. "
            f"Expected {expected_count}, "
            f"found {points_count}."
        )


    logger.info(
        "Qdrant collection validation passed."
    )


# ============================================================
# RETRIEVAL SMOKE TESTS
# ============================================================

def run_retrieval_smoke_tests(
    embedding_service:
        GeminiEmbeddingService,
) -> None:

    """
    Immediately test the new collection after ingestion.

    Several of these are the exact concepts that failed in the
    old collection.
    """

    test_cases = [

        {
            "question":
                "What happens when an employee resigns?",

            "expected_policy":
                "Separation",
        },

        {
            "question":
                (
                    "What rules apply to "
                    "international business travel?"
                ),

            "expected_policy":
                "International Travel",
        },

        {
            "question":
                (
                    "What health insurance "
                    "benefits are available?"
                ),

            "expected_policy":
                "Health & Insurance",
        },

        {
            "question":
                (
                    "What happens to my outstanding "
                    "loan if I resign?"
                ),

            "expected_policy":
                "Loans & Advances",
        },

        {
            "question":
                (
                    "Can the company pay for my "
                    "professional certification?"
                ),

            "expected_policy":
                (
                    "Certification & "
                    "Professional Membership"
                ),
        },

        {
            "question":
                (
                    "How many hours should "
                    "employees normally work?"
                ),

            "expected_policy":
                "Working Hours",
        },

        {
            "question":
                "What is the leave policy?",

            "expected_policy":
                "Leave",
        },

        {
            "question":
                "Can I work flexible hours?",

            "expected_policy":
                "Flexible Working",
        },

        {
            "question":
                "What is the social media policy?",

            "expected_policy":
                "Social Media",
        },

        {
            "question":
                (
                    "Can I attend an external "
                    "professional conference?"
                ),

            "expected_policy":
                "Seminars & Conferences",
        },
    ]


    print()
    print("=" * 80)
    print("RETRIEVAL SMOKE TESTS")
    print("=" * 80)


    passed = 0


    for number, test in enumerate(
        test_cases,
        start=1,
    ):

        query_vector = (
            embedding_service
            .embed_query(
                test["question"]
            )
        )


        response = (
            qdrant_client
            .query_points(

                collection_name=
                    COLLECTION_NAME,

                query=
                    query_vector,

                limit=
                    10,

                with_payload=
                    True,
            )
        )


        retrieved_policies = [

            (
                point.payload
                or {}
            ).get(
                "policy",
                "",
            )

            for point
            in response.points
        ]


        expected_policy = (
            test[
                "expected_policy"
            ]
        )


        found = (

            expected_policy

            in retrieved_policies
        )


        if found:

            passed += 1

            result = "PASS"

        else:

            result = "FAIL"


        print()

        print(
            f"{number:02d}. "
            f"{result} | "
            f"{test['question']}"
        )


        print(
            f"Expected: "
            f"{expected_policy}"
        )


        print(
            "Top 5:"
        )


        for rank, point in enumerate(
            response.points[:5],
            start=1,
        ):

            payload = (
                point.payload
                or {}
            )


            print(
                f"  {rank}. "
                f"{payload.get('category', 'Unknown')} "
                f"> "
                f"{payload.get('policy', 'Unknown')} "
                f"| "
                f"{payload.get('subsection') or 'General'} "
                f"| "
                f"{point.score:.4f}"
            )


    total = len(
        test_cases
    )


    recall = (

        passed / total

        if total

        else 0.0
    )


    print()
    print("=" * 80)
    print("SMOKE TEST RESULT")
    print("=" * 80)

    print(
        f"Passed:    {passed}/{total}"
    )

    print(
        f"Recall@10: {recall:.1%}"
    )


    if passed == total:

        logger.info(
            "All retrieval smoke tests passed."
        )

    else:

        logger.warning(
            "Some retrieval smoke tests failed | "
            "passed=%d/%d",
            passed,
            total,
        )


# ============================================================
# SAMPLE CHUNK OUTPUT
# ============================================================

def print_sample_chunks(
    chunks: List[PolicyChunk],
    count: int = 3,
) -> None:

    print()
    print("=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)


    for chunk in chunks[:count]:

        print()

        print(
            f"Policy: "
            f"{chunk.category} > "
            f"{chunk.policy}"
        )

        print(
            f"Policy ID: "
            f"{chunk.policy_id}"
        )

        print(
            f"Subsection: "
            f"{chunk.subsection or 'General'}"
        )

        print(
            f"Chunk index: "
            f"{chunk.chunk_index}"
        )

        print()

        print(
            chunk.text[:800]
        )

        print()
        print(
            "-" * 80
        )


# ============================================================
# INGESTION SUMMARY
# ============================================================

def print_ingestion_summary(
    policies: List[PolicySection],
    chunks: List[PolicyChunk],
    vector_size: int,
) -> None:

    categories = sorted({

        policy.category

        for policy
        in policies
    })


    print()
    print("=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)

    print(
        f"Source:               "
        f"{DOCUMENT_PATH}"
    )

    print(
        f"Document version:     "
        f"{DOCUMENT_VERSION}"
    )

    print(
        f"Collection:           "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Embedding model:      "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Embedding dimension:  "
        f"{vector_size}"
    )

    print(
        f"Embedding batch size: "
        f"{EMBEDDING_BATCH_SIZE}"
    )

    print(
        f"Categories:           "
        f"{len(categories)}"
    )

    print(
        f"Policies:             "
        f"{len(policies)}"
    )

    print(
        f"Chunks:               "
        f"{len(chunks)}"
    )

    print()

    print(
        "Policies by category:"
    )


    for category in categories:

        policy_count = sum(

            1

            for policy
            in policies

            if policy.category
            == category
        )


        chunk_count = sum(

            1

            for chunk
            in chunks

            if chunk.category
            == category
        )


        print(
            f"  - {category}: "
            f"{policy_count} policies, "
            f"{chunk_count} chunks"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("ENTERPRISE POLICY RAG INGESTION V2")
    print("=" * 80)

    print()

    print(
        f"Document:   "
        f"{DOCUMENT_PATH}"
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Embedding:  "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Batch size: "
        f"{EMBEDDING_BATCH_SIZE}"
    )

    print()


    # ========================================================
    # STEP 1/8
    # PARSE MARKDOWN
    # ========================================================

    logger.info(
        "STEP 1/8 | Parsing Markdown"
    )


    policies = (
        parse_markdown_document(
            DOCUMENT_PATH
        )
    )


    # ========================================================
    # STEP 2/8
    # VALIDATE POLICIES
    # ========================================================

    logger.info(
        "STEP 2/8 | Validating policies"
    )


    validate_policies(
        policies
    )


    # ========================================================
    # STEP 3/8
    # CREATE POLICY-AWARE CHUNKS
    # ========================================================

    logger.info(
        "STEP 3/8 | Creating policy-aware chunks"
    )


    chunks = (
        create_policy_chunks(
            policies
        )
    )


    # ========================================================
    # STEP 4/8
    # VALIDATE CHUNKS
    # ========================================================

    logger.info(
        "STEP 4/8 | Validating chunks"
    )


    validate_chunks(
        chunks
    )


    # Show samples before spending embedding quota.

    print_sample_chunks(
        chunks
    )


    # ========================================================
    # STEP 5/8
    # GENERATE EMBEDDINGS
    # ========================================================

    logger.info(
        "STEP 5/8 | Generating embeddings"
    )


    embedding_service = (
        GeminiEmbeddingService(

            client=
                gemini_client,

            model=
                EMBEDDING_MODEL,

            batch_size=
                EMBEDDING_BATCH_SIZE,

            retries=
                EMBEDDING_MAX_RETRIES,
        )
    )


    embedding_inputs = [

        chunk.embedding_text

        for chunk
        in chunks
    ]


    embeddings = (
        embedding_service
        .embed_documents(
            embedding_inputs
        )
    )


    # ========================================================
    # STEP 6/8
    # VALIDATE EMBEDDINGS
    # ========================================================

    logger.info(
        "STEP 6/8 | Validating embeddings"
    )


    vector_size = (
        validate_embeddings(

            embeddings=
                embeddings,

            expected_count=
                len(chunks),
        )
    )


    # ========================================================
    # STEP 7/8
    # CREATE V2 COLLECTION + UPLOAD
    # ========================================================

    logger.info(
        "STEP 7/8 | Creating and populating Qdrant V2"
    )


    # IMPORTANT:
    #
    # Collection recreation happens ONLY after all embeddings
    # have been generated and validated.
    #
    # Therefore a Gemini failure cannot leave us with a
    # half-created / half-populated V2 collection.

    recreate_v2_collection(

        vector_size=
            vector_size
    )


    upload_to_qdrant(

        chunks=
            chunks,

        embeddings=
            embeddings,
    )


    validate_qdrant_collection(

        expected_count=
            len(chunks)
    )


    # ========================================================
    # STEP 8/8
    # RETRIEVAL SMOKE TESTS
    # ========================================================

    logger.info(
        "STEP 8/8 | Running retrieval smoke tests"
    )


    run_retrieval_smoke_tests(

        embedding_service=
            embedding_service
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_ingestion_summary(

        policies=
            policies,

        chunks=
            chunks,

        vector_size=
            vector_size,
    )


    # ========================================================
    # NEXT STEP
    # ========================================================

    print()
    print("=" * 80)
    print("NEXT STEP")
    print("=" * 80)

    print()

    print(
        "Your original Qdrant collection has NOT "
        "been modified."
    )

    print()

    print(
        "If the retrieval smoke tests above are good:"
    )

    print()

    print(
        "1. Change rag.py:"
    )

    print()

    print(
        '   COLLECTION_NAME = '
        '"acme_employee_policies_v2"'
    )

    print()

    print(
        "2. Run:"
    )

    print()

    print(
        "   python eval.py"
    )

    print()

    print(
        "3. Compare the new Recall@10 against "
        "the old baseline:"
    )

    print()

    print(
        "   OLD Recall@10 = 70%"
    )

    print()

    print(
        "Do not delete the original collection "
        "until V2 has been fully validated."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()