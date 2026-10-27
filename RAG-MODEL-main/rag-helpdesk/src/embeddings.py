from __future__ import annotations

import logging
import random
import time
from typing import Callable, List, Optional, TypeVar

import requests

try:
    from google import genai
    from google.genai import types

    _HAS_GOOGLE_GENAI = True
except ImportError:  # pragma: no cover
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GOOGLE_GENAI = False

try:
    from openai import OpenAI as OpenAIClient, APIStatusError

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    OpenAIClient = None  # type: ignore
    APIStatusError = Exception  # type: ignore
    _HAS_OPENAI = False

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_BASE_DELAY,
    GE_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ======================================================================
# ERROR / RETRY HELPERS
# ======================================================================

class RateLimitError(RuntimeError):
    """Raised when embedding API rate limits are exceeded."""


def _is_rate_limit_error(exc: Exception) -> bool:
    if _HAS_OPENAI and isinstance(exc, APIStatusError):
        if getattr(exc, "status_code", None) == 429:
            return True

    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)

        if response is not None and response.status_code == 429:
            return True

    message = str(exc).lower()

    return (
        "429" in message
        or "rate limit" in message
        or "quota" in message
        or "resource exhausted" in message
        or "too many requests" in message
    )


def _get_retry_after_seconds(
    exc: Exception,
) -> Optional[float]:

    if _HAS_OPENAI and isinstance(exc, APIStatusError):

        response = getattr(exc, "response", None)

        headers = (
            getattr(response, "headers", None)
            if response is not None
            else None
        )

        if headers:

            retry_after = (
                headers.get("retry-after")
                or headers.get("Retry-After")
            )

            if retry_after is not None:

                try:
                    return float(retry_after)

                except (TypeError, ValueError):
                    pass

    if isinstance(exc, requests.HTTPError):

        response = getattr(exc, "response", None)

        if response is not None:

            retry_after = (
                response.headers.get("Retry-After")
                or response.headers.get("retry-after")
            )

            if retry_after is not None:

                try:
                    return float(retry_after)

                except (TypeError, ValueError):
                    pass

    return None


def _execute_with_retry(
    operation_name: str,
    func: Callable[[], T],
    *,
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
) -> T:

    retries = (
        max_retries
        if max_retries is not None
        else EMBEDDING_MAX_RETRIES
    )

    delay_base = (
        base_delay
        if base_delay is not None
        else EMBEDDING_RETRY_BASE_DELAY
    )

    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):

        try:
            return func()

        except Exception as exc:

            if not _is_rate_limit_error(exc):
                raise

            last_exc = exc

            if attempt >= retries:
                break

            retry_after = _get_retry_after_seconds(exc)

            delay = (
                retry_after
                if retry_after is not None
                else delay_base * (2 ** attempt)
            )

            delay += random.uniform(0, 0.5)

            logger.warning(
                "Rate limit on %s, attempt %d/%d. "
                "Retrying in %.1fs.",
                operation_name,
                attempt + 1,
                retries,
                delay,
            )

            time.sleep(delay)

    raise RateLimitError(
        f"Rate limit exceeded after {retries} retries "
        f"during {operation_name}: {last_exc}"
    ) from last_exc


# ======================================================================
# EMBEDDING SERVICE
# ======================================================================

class EmbeddingService:
    """
    Flexible embedding service.

    Supported modes:

    1. Gemini:
       EMBEDDING_MODEL=gemini-embedding-2

       Uses Google's native google-genai SDK.

    2. Local:
       EMBEDDING_MODEL=local:BAAI/bge-small-en-v1.5

       Uses Sentence Transformers locally.

       IMPORTANT:
       Sentence Transformers is imported ONLY when local mode
       is selected.

    3. OpenAI-compatible:
       Any other embedding model.

       Uses the existing OpenAI-compatible implementation.
    """

    # Gemini native batch size.
    GEMINI_BATCH_SIZE = 50

    # Local embedding batch size.
    LOCAL_BATCH_SIZE = 32

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):

        self.api_key = (
            api_key
            or GE_API_KEY
            or LLM_API_KEY
        )

        self.model = (
            model
            or EMBEDDING_MODEL
        )

        self.base_url = (
            base_url
            or LLM_BASE_URL
            or "https://generativelanguage.googleapis.com/v1beta/openai"
        ).rstrip("/")

        self._google_client = None
        self._client: Optional["OpenAIClient"] = None
        self._local_model = None

        self.dimension: Optional[int] = None

        self._validate_model_endpoint_pair()

        # ==============================================================
        # LOCAL EMBEDDING MODE
        # ==============================================================

        if self.model.startswith("local:"):

            local_model_name = (
                self.model.replace("local:", "", 1)
                .strip()
            )

            if not local_model_name:

                raise ValueError(
                    "Local embedding model name is missing. "
                    "Example: local:BAAI/bge-small-en-v1.5"
                )

            try:

                # IMPORTANT:
                # This import happens ONLY in local mode.
                from sentence_transformers import (
                    SentenceTransformer
                )

            except ImportError as exc:

                raise RuntimeError(
                    "Local embeddings require "
                    "sentence-transformers.\n\n"
                    "Install it with:\n"
                    "pip install sentence-transformers"
                ) from exc

            try:

                logger.info(
                    "Loading LOCAL embedding model: %s",
                    local_model_name,
                )

                self._local_model = (
                    SentenceTransformer(
                        local_model_name,
                        local_files_only=True,
                    )
                )

                self.dimension = (
                    self._local_model
                    .get_sentence_embedding_dimension()
                )

                logger.info(
                    "Local embedding model loaded "
                    "(dimension=%s).",
                    self.dimension,
                )

                print(
                    f"LOCAL EMBEDDINGS ENABLED: "
                    f"{local_model_name}"
                )

                print(
                    f"Embedding dimension: "
                    f"{self.dimension}"
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Failed to load local embedding model "
                    f"'{local_model_name}': {exc}"
                ) from exc

            # IMPORTANT:
            #
            # Stop initialization here.
            #
            # No Gemini client.
            # No OpenAI client.
            # No embedding API calls.

            return

        # ==============================================================
        # GEMINI NATIVE EMBEDDING MODE
        # ==============================================================

        if (
            self.model.startswith("gemini-embedding")
            and _HAS_GOOGLE_GENAI
            and self.api_key
        ):

            try:

                self._google_client = (
                    genai.Client(
                        api_key=self.api_key
                    )
                )

                logger.info(
                    "Native Google GenAI embedding client "
                    "enabled (model=%s).",
                    self.model,
                )

                print(
                    f"GEMINI EMBEDDINGS ENABLED: "
                    f"{self.model}"
                )

            except Exception as exc:

                logger.error(
                    "Failed to initialize Google GenAI "
                    "client: %s",
                    exc,
                )

                self._google_client = None

        # ==============================================================
        # EXISTING OPENAI-COMPATIBLE MODE
        # ==============================================================

        if (
            not self.model.startswith("gemini-embedding")
            and _HAS_OPENAI
            and self.api_key
            and self.base_url
        ):

            try:

                import httpx

                http_client = httpx.Client(
                    verify=False,
                    timeout=httpx.Timeout(
                        20.0,
                        connect=6.0,
                    ),
                )

                try:

                    self._client = (
                        OpenAIClient(
                            api_key=self.api_key,
                            base_url=self.base_url,
                            default_headers={
                                "x-api-key": self.api_key
                            },
                            http_client=http_client,
                        )
                    )

                except TypeError:

                    self._client = (
                        OpenAIClient(
                            api_key=self.api_key,
                            base_url=self.base_url,
                            http_client=http_client,
                        )
                    )

            except Exception as exc:

                logger.error(
                    "Failed to initialize OpenAI client: %s",
                    exc,
                )

                self._client = None

    def _validate_model_endpoint_pair(self) -> None:
        """Apply only the checks that are actually supported by this endpoint.

        The Capgemini OpenAI-compatible endpoint is known to accept the Titan
        embedding model, so rejecting it here causes a false negative and blocks
        real semantic embeddings.
        """
        model_name = (self.model or "").lower()
        base = (self.base_url or "").lower()

        if not model_name or not base:
            return

        has_capgemini_openai_endpoint = (
            "openai.generative.engine.capgemini.com" in base
            or "generative.engine.capgemini.com" in base
        )
        is_aws_titan_embedding = "amazon.titan-embed" in model_name

        # Verified against the live Capgemini endpoint: Titan embeddings are
        # allowed there. Do not raise an incompatibility error for this pair.
        if has_capgemini_openai_endpoint and is_aws_titan_embedding:
            logger.info(
                "Capgemini OpenAI-compatible endpoint accepted Titan embedding model '%s'.",
                self.model,
            )
            return

    # ==================================================================
    # SINGLE EMBEDDING
    # ==================================================================

    def create_embedding(
        self,
        text: str,
    ) -> Optional[List[float]]:

        if not text or not text.strip():

            logger.warning(
                "Cannot create embedding for empty text."
            )

            return None

        # ==============================================================
        # LOCAL
        # ==============================================================

        if self.model.startswith("local:"):

            return self._local_embed_single(
                text
            )

        # ==============================================================
        # GEMINI
        # ==============================================================

        if self.model.startswith(
            "gemini-embedding"
        ):

            return self._google_embed_single(
                text
            )

        # ==============================================================
        # OPENAI-COMPATIBLE
        # ==============================================================

        if not self.api_key or not self.model:

            logger.warning(
                "Embedding configuration incomplete."
            )

            return None

        if self._client is not None:

            try:

                resp = _execute_with_retry(
                    "single embedding",
                    lambda: (
                        self._client
                        .embeddings
                        .create(
                            model=self.model,
                            input=text,
                        )
                    ),
                    max_retries=2,
                    base_delay=0.5,
                )

                return resp.data[0].embedding

            except RateLimitError:

                raise

            except Exception as exc:

                logger.error(
                    "OpenAI client embedding error "
                    "(model=%s): %s",
                    self.model,
                    exc,
                )
                raise RuntimeError(
                    f"Embedding model '{self.model}' failed on the configured endpoint "
                    f"'{self.base_url}'. Check the API key, model access, and provider permissions."
                ) from exc

        try:
            return self._requests_embed_single(text)
        except Exception as exc:
            logger.error(
                "Raw embedding request failed for model '%s' on '%s': %s",
                self.model,
                self.base_url,
                exc,
            )
            raise RuntimeError(
                f"Embedding model '{self.model}' failed on the configured endpoint "
                f"'{self.base_url}'. Check the API key, model access, and provider permissions."
            ) from exc

    # ==================================================================
    # BATCH EMBEDDING
    # ==================================================================

    def create_embeddings_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        if not texts:
            return []

        # ==============================================================
        # LOCAL
        # ==============================================================

        if self.model.startswith("local:"):

            return self._local_embed_batch(
                texts
            )

        # ==============================================================
        # GEMINI
        # ==============================================================

        if self.model.startswith(
            "gemini-embedding"
        ):

            return self._google_embed_batch(
                texts
            )

        # ==============================================================
        # OPENAI-COMPATIBLE
        # ==============================================================

        if not self.api_key or not self.model:

            raise ValueError(
                "Embedding configuration is incomplete."
            )

        if self._client is not None:

            try:
                return self._client_embed_batch(texts)
            except Exception as exc:
                logger.error(
                    "Remote batch embedding failed for model '%s' on '%s': %s",
                    self.model,
                    self.base_url,
                    exc,
                )
                raise RuntimeError(
                    f"Embedding model '{self.model}' failed on the configured endpoint "
                    f"'{self.base_url}'. Check the API key, model access, and provider permissions."
                ) from exc

        try:
            return self._requests_embed_batch(texts)
        except Exception as exc:
            logger.error(
                "Remote batch requests failed for model '%s' on '%s': %s",
                self.model,
                self.base_url,
                exc,
            )
            raise RuntimeError(
                f"Embedding model '{self.model}' failed on the configured endpoint "
                f"'{self.base_url}'. Check the API key, model access, and provider permissions."
            ) from exc

    # ==================================================================
    # LOCAL EMBEDDINGS
    # ==================================================================

    def _local_embed_single(
        self,
        text: str,
    ) -> List[float]:

        if self._local_model is None:

            raise RuntimeError(
                "Local embedding model is not loaded."
            )

        try:

            embedding = (
                self._local_model.encode(
                    text,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            )

            return embedding.tolist()

        except Exception as exc:

            logger.error(
                "Local embedding failed: %s",
                exc,
            )

            raise RuntimeError(
                "Failed to create local embedding."
            ) from exc

    def _local_embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        if self._local_model is None:

            raise RuntimeError(
                "Local embedding model is not loaded."
            )

        try:

            embeddings = (
                self._local_model.encode(
                    texts,
                    batch_size=self.LOCAL_BATCH_SIZE,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )
            )

            result = embeddings.tolist()

            if len(result) != len(texts):

                raise RuntimeError(
                    f"Local model returned "
                    f"{len(result)} embeddings for "
                    f"{len(texts)} inputs."
                )

            return result

        except Exception as exc:

            logger.error(
                "Local batch embedding failed: %s",
                exc,
            )

            raise RuntimeError(
                "Failed to create local batch embeddings."
            ) from exc

    # ==================================================================
    # GEMINI - SINGLE
    # ==================================================================

    def _google_embed_single(
        self,
        text: str,
    ) -> Optional[List[float]]:

        if self._google_client is None:

            raise RuntimeError(
                "google-genai is not installed or the "
                "Google GenAI client could not be initialized."
            )

        try:

            result = _execute_with_retry(
                "Gemini single embedding",
                lambda: (
                    self._google_client
                    .models
                    .embed_content(
                        model=self.model,
                        contents=text,
                        config=(
                            types.EmbedContentConfig(
                                output_dimensionality=3072
                            )
                        ),
                    )
                ),
            )

            if not result.embeddings:

                raise RuntimeError(
                    "Gemini returned no embeddings."
                )

            values = (
                result
                .embeddings[0]
                .values
            )

            if values is None:

                raise RuntimeError(
                    "Gemini returned an embedding "
                    "without values."
                )

            return list(values)

        except RateLimitError:

            raise

        except Exception as exc:

            logger.error(
                "Gemini embedding error "
                "(model=%s): %s",
                self.model,
                exc,
            )

            return None

    # ==================================================================
    # GEMINI - BATCH
    # ==================================================================

    def _google_embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        if self._google_client is None:

            raise RuntimeError(
                "google-genai is not installed or the "
                "Google GenAI client could not be initialized."
            )

        all_embeddings: List[List[float]] = []

        total = len(texts)

        for start in range(
            0,
            total,
            self.GEMINI_BATCH_SIZE,
        ):

            batch = texts[
                start:
                start + self.GEMINI_BATCH_SIZE
            ]

            batch_number = (
                start // self.GEMINI_BATCH_SIZE
            ) + 1

            total_batches = (
                total
                + self.GEMINI_BATCH_SIZE
                - 1
            ) // self.GEMINI_BATCH_SIZE

            logger.info(
                "Gemini embedding batch %d/%d "
                "(%d texts).",
                batch_number,
                total_batches,
                len(batch),
            )

            # Each Content object represents ONE input.
            contents = [
                types.Content(
                    parts=[
                        types.Part.from_text(
                            text=text
                        )
                    ]
                )
                for text in batch
            ]

            # Small pause between API requests.
            #
            # This is only used for Gemini.
            # Local embeddings do not sleep.
            if all_embeddings:

                time.sleep(1.0)

            try:

                result = _execute_with_retry(
                    (
                        f"Gemini batch embedding "
                        f"({len(batch)} items)"
                    ),
                    lambda: (
                        self._google_client
                        .models
                        .embed_content(
                            model=self.model,
                            contents=contents,
                            config=(
                                types.EmbedContentConfig(
                                    output_dimensionality=3072
                                )
                            ),
                        )
                    ),
                    max_retries=3,
                    base_delay=2.0,
                )

            except RateLimitError:

                raise

            except Exception as exc:

                logger.error(
                    "Gemini batch embedding failed: %s",
                    exc,
                )

                raise RuntimeError(
                    "Gemini batch embedding failed "
                    f"for batch "
                    f"{batch_number}/{total_batches}"
                ) from exc

            if not result.embeddings:

                raise RuntimeError(
                    "Gemini returned no embeddings "
                    f"for batch {batch_number}."
                )

            batch_embeddings = []

            for embedding in result.embeddings:

                values = embedding.values

                if values is None:

                    raise RuntimeError(
                        "Gemini returned an embedding "
                        "without values."
                    )

                batch_embeddings.append(
                    list(values)
                )

            if (
                len(batch_embeddings)
                != len(batch)
            ):

                raise RuntimeError(
                    "Gemini returned "
                    f"{len(batch_embeddings)} embeddings "
                    f"for {len(batch)} inputs."
                )

            all_embeddings.extend(
                batch_embeddings
            )

            logger.info(
                "Gemini progress: %d/%d embeddings.",
                len(all_embeddings),
                total,
            )

        return all_embeddings

    # ==================================================================
    # EXISTING OPENAI SDK BATCH
    # ==================================================================

    def _client_embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        try:

            resp = _execute_with_retry(
                f"batch embedding "
                f"({len(texts)} items)",
                lambda: (
                    self._client
                    .embeddings
                    .create(
                        model=self.model,
                        input=texts,
                    )
                ),
            )

            results = [None] * len(texts)

            for item in resp.data:

                if (
                    item.index is not None
                    and 0 <= item.index < len(texts)
                ):

                    results[
                        item.index
                    ] = item.embedding

            none_indices = [
                i
                for i, value
                in enumerate(results)
                if value is None
            ]

            if none_indices:

                logger.warning(
                    "Batch embed: %d items had "
                    "no index; falling back per-item.",
                    len(none_indices),
                )

                for i in none_indices:

                    emb = (
                        self._client_embed_single(
                            texts[i]
                        )
                    )

                    if emb is None:

                        raise RuntimeError(
                            f"Embedding failed for "
                            f"item {i}/"
                            f"{len(texts)}"
                        )

                    results[i] = emb

            return results  # type: ignore

        except RateLimitError:

            raise

        except Exception as exc:

            if _is_rate_limit_error(exc):

                raise RateLimitError(
                    "Rate limit exceeded during "
                    f"batch embedding: {exc}"
                ) from exc

            logger.warning(
                "Batch embedding failed (%s); "
                "retrying per-item.",
                exc,
            )

            embeddings: List[List[float]] = []

            for i, text in enumerate(texts):

                emb = (
                    self._client_embed_single(
                        text
                    )
                )

                if emb is None:

                    raise RuntimeError(
                        f"Embedding failed for "
                        f"item {i + 1}/"
                        f"{len(texts)}"
                    ) from exc

                embeddings.append(emb)

            return embeddings

    # ==================================================================
    # EXISTING OPENAI SDK SINGLE
    # ==================================================================

    def _client_embed_single(
        self,
        text: str,
    ) -> Optional[List[float]]:

        try:

            resp = _execute_with_retry(
                "single embedding (client)",
                lambda: (
                    self._client
                    .embeddings
                    .create(
                        model=self.model,
                        input=text,
                    )
                ),
            )

            return resp.data[0].embedding

        except RateLimitError:

            raise

        except Exception as exc:

            logger.error(
                "OpenAI client embedding error "
                "(model=%s): %s",
                self.model,
                exc,
            )

            return None

    # ==================================================================
    # RAW REQUESTS
    # ==================================================================

    def _embeddings_endpoint(self) -> str:

        return (
            f"{self.base_url}/embeddings"
        )

    def _requests_embed_single(
        self,
        text: str,
    ) -> Optional[List[float]]:

        headers = {
            "Authorization":
                f"Bearer {self.api_key}",

            "x-api-key":
                self.api_key,

            "Content-Type":
                "application/json",
        }

        payload = {
            "model": self.model,
            "input": text,
        }

        try:

            resp = _execute_with_retry(
                "single embedding (requests)",
                lambda: (
                    self._post_embedding_request(
                        headers,
                        payload,
                        timeout=30,
                    )
                ),
            )

            return (
                resp.json()
                ["data"][0]
                ["embedding"]
            )

        except RateLimitError:

            raise

        except Exception as exc:

            logger.error(
                "Embedding request failed: %s",
                exc,
            )

            return None

    def _requests_embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        headers = {
            "Authorization":
                f"Bearer {self.api_key}",

            "x-api-key":
                self.api_key,

            "Content-Type":
                "application/json",
        }

        try:

            resp = _execute_with_retry(
                (
                    f"batch embedding "
                    f"(requests, {len(texts)} items)"
                ),
                lambda: (
                    self._post_embedding_request(
                        headers,
                        {
                            "model":
                                self.model,
                            "input":
                                texts,
                        },
                        timeout=60,
                    )
                ),
            )

            items = (
                resp
                .json()
                .get("data", [])
            )

            if len(items) != len(texts):

                raise RuntimeError(
                    "Batch response length "
                    f"{len(items)} != request size "
                    f"{len(texts)}"
                )

            return [
                item["embedding"]
                for item in items
            ]

        except RateLimitError:

            raise

        except Exception as exc:

            if _is_rate_limit_error(exc):

                raise RateLimitError(
                    "Rate limit exceeded during "
                    f"batch embedding: {exc}"
                ) from exc

            logger.warning(
                "Batch embed (requests) failed "
                "(%s); retrying per-item.",
                exc,
            )

            results: List[List[float]] = []

            for i, text in enumerate(texts):

                emb = (
                    self._requests_embed_single(
                        text
                    )
                )

                if emb is None:

                    raise RuntimeError(
                        f"Embedding failed for "
                        f"item {i + 1}/"
                        f"{len(texts)}"
                    ) from exc

                results.append(emb)

            return results

    # ==================================================================
    # HTTP HELPER
    # ==================================================================

    def _post_embedding_request(
        self,
        headers: dict[str, str],
        payload: dict,
        *,
        timeout: int,
    ) -> requests.Response:

        resp = requests.post(
            self._embeddings_endpoint(),
            headers=headers,
            json=payload,
            timeout=timeout,
            verify=False,
        )

        try:

            resp.raise_for_status()

        except requests.HTTPError as exc:

            if resp.status_code == 429:
                raise exc

            raise

        return resp

    # ==================================================================
    # QUERY COMPATIBILITY
    # ==================================================================

    def embed_query(
        self,
        text: str,
    ) -> List[float]:

        embedding = self.create_embedding(
            text
        )

        if embedding is None:

            raise RuntimeError(
                "Failed to generate query embedding."
            )

        return embedding


# ======================================================================
# SYNTHETIC EMBEDDINGS
# ======================================================================

class SyntheticEmbeddingService:
    """
    Deterministic synthetic embeddings for local
    development/testing without API calls.

    This is NOT recommended for actual semantic retrieval.
    """

    DIMS = 768

    def __init__(self) -> None:

        self.model = "synthetic"

        logger.info(
            "SyntheticEmbeddingService active â€” "
            "no API calls will be made."
        )

    def create_embedding(
        self,
        text: str,
    ) -> List[float]:

        import hashlib
        import math

        seed = int(
            hashlib.md5(
                text.encode(),
                usedforsecurity=False,
            ).hexdigest(),
            16,
        )

        vals: List[float] = []

        for i in range(self.DIMS):

            angle = (
                seed
                + i * 31337
            ) % 360

            vals.append(
                math.sin(
                    math.radians(angle)
                )
            )

        norm = math.sqrt(
            sum(v * v for v in vals)
        ) or 1.0

        return [
            v / norm
            for v in vals
        ]

    def create_embeddings_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        return [
            self.create_embedding(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> List[float]:

        return self.create_embedding(text)
