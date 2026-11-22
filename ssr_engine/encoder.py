"""Text to vectors: a sentence-embedding model on the CPU, or a deterministic stand-in.

The model is chosen by the Space's variables, never by a request. With none set, the Space runs
google/embeddinggemma-2; there is no fallback to the stand-in, so a model that fails to load stops
the Space instead of quietly scoring with something else.

    SSR_ENGINE_ENCODER   model (default) or hash
    SSR_ENGINE_MODEL     a sentence-transformers model id; default google/embeddinggemma-2
    SSR_ENGINE_PREFIX    text put in front of every input; defaults to the model's sentence-similarity
                         prefix for EmbeddingGemma and to nothing for any other model

Weights run in float32 only. EmbeddingGemma's model card says not to use float16: its activations
exceed that range and the vectors come back wrong without an error. Anything but float32 is refused
before a model is touched, and what comes back is checked for values that are not finite.

Nothing here keeps a text or a vector after the call that asked for it returns.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading

DEFAULT_MODEL = "google/embeddinggemma-2"
SIMILARITY_PREFIX = "task: sentence similarity | query: "
HASH_DIMENSIONS = 512
_WORD = re.compile(r"[a-z']+")
_STOP = frozenset("a an the and or but of to in on for with at by from as is are was were be been it its this that these those i we you they he she "
                  "my our your their me us them do does did have has had would could should will can may might not no so if then than too very "
                  "just about into over up out off there here what which who whom how why when where am".split())


def check_dtype(dtype) -> str:
    name = str(dtype or "").strip().lower().replace("torch.", "")
    if name in ("float16", "half", "fp16"):
        raise ValueError("float16 is refused: the encoder's activations exceed its range and the vectors come back wrong. Use float32.")
    if name != "float32":
        raise ValueError("Unsupported dtype: the encoder runs in float32 only.")
    return name


def finite(vectors) -> list[list[float]]:
    out = [[float(x) for x in vector] for vector in vectors]
    if any(not math.isfinite(x) for vector in out for x in vector):
        raise ValueError("The encoder returned a value that is not finite; the vectors cannot be trusted.")
    return out


class HashEncoder:
    """Hashed bag of words and word pairs, unit length. Deterministic and offline: texts that share
    wording come out close. It says nothing about meaning; it exists for tests and for trying the
    interface without downloading a model."""

    name = "hash"
    dtype = "float32"
    prefix = ""
    meaningful = False

    def encode(self, texts) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = [w for w in _WORD.findall(str(text).lower()) if w not in _STOP]
            vector = [0.0] * HASH_DIMENSIONS
            for token in words + [a + " " + b for a, b in zip(words, words[1:], strict=False)]:
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                vector[int.from_bytes(digest[:4], "big") % HASH_DIMENSIONS] += 1.0 if digest[4] % 2 else -1.0
            norm = math.sqrt(sum(x * x for x in vector))
            vectors.append([x / norm for x in vector] if norm else vector)
        return vectors


class SentenceEncoder:
    """A sentence-transformers model on the CPU in float32. Its libraries are imported only here."""

    meaningful = True

    def __init__(self, model_id: str = DEFAULT_MODEL, dtype: str = "float32", prefix: str | None = None, factory=None):
        self.dtype = check_dtype(dtype)          # before any import or download
        self.name = model_id
        self.prefix = (SIMILARITY_PREFIX if "embeddinggemma" in model_id.lower() else "") if prefix is None else prefix
        self._lock = threading.Lock()            # one request in the model at a time: no batch ever mixes two callers' texts
        if factory is not None:                  # tests inject a stand-in for the library's model
            self.model = factory()
            return
        import torch
        from sentence_transformers import SentenceTransformer
        # float32 is asked for by name. Left to itself the library loads the precision the model's own
        # config names, which for EmbeddingGemma 2 is bfloat16.
        precision = {"dtype": torch.float32}
        try:                                     # text encoder only, where the model and library allow it
            self.model = SentenceTransformer(model_id, device="cpu", model_kwargs=precision, config_kwargs={"vision_config": None, "audio_config": None})
        except Exception:  # noqa: BLE001 - whatever a model makes of those two settings, the plain load is tried once
            self.model = SentenceTransformer(model_id, device="cpu", model_kwargs=precision)
        # What matters is what was actually loaded, so that is checked: anything else stops the Space instead of scoring.
        wrong = sorted({str(p.dtype) for p in self.model.parameters() if p.is_floating_point() and p.dtype != torch.float32})
        if wrong:
            raise ValueError("The encoder loaded with weights that are not float32.")

    def encode(self, texts) -> list[list[float]]:
        with self._lock:
            vectors = self.model.encode([self.prefix + str(t) for t in texts], normalize_embeddings=True, show_progress_bar=False)
        return finite(vectors.tolist() if hasattr(vectors, "tolist") else vectors)


_ENCODER = None
_LOAD = threading.Lock()


def get():
    """The Space's one encoder, loaded on first use. It holds weights, never anything a caller sent."""
    global _ENCODER
    with _LOAD:
        if _ENCODER is None:
            kind = os.getenv("SSR_ENGINE_ENCODER", "model").strip().lower()
            if kind == "hash":
                _ENCODER = HashEncoder()
            elif kind == "model":
                _ENCODER = SentenceEncoder(os.getenv("SSR_ENGINE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
                                           "float32", os.getenv("SSR_ENGINE_PREFIX"))
            else:
                raise ValueError("SSR_ENGINE_ENCODER must be model or hash.")
        return _ENCODER
