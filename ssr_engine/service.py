"""The engine's operations as plain functions: a request in, a result out, nothing kept.

    about()              which encoder is running, and the request limits
    score(request)       reactions -> a distribution over the five-point scale and its mean, per reaction
    calibrate(request)   reactions with their intended point -> the temperature that fits each mapping

Rules this module holds to, because the Space is public and callers do not know each other:

  * A result never repeats a text. Reactions are identified by their position in the request.
  * An error never repeats a text either: messages name a position and a limit, nothing else.
  * Nothing a caller sends is stored, cached or logged. The only cache is the default anchors'
    vectors, which no caller supplied. Anchors sent with a request are embedded for that request only.
  * Requests are bounded (LIMITS), so one caller cannot occupy the encoder for long.
"""
from __future__ import annotations

import threading
import unicodedata

from . import anchors as default_anchors
from . import calibration
from . import encoder as encoders
from . import mapping

VERSION = "1.0.0"
LIMITS = {"texts": 200, "text_chars": 2000, "anchor_sets": 6, "anchor_chars": 400, "labelled": 400, "decimals": 6}


class EngineError(ValueError):
    """A request the engine refuses. The message is safe to show and to log: it never carries request content."""


_DEFAULT_VECTORS: dict[str, list] = {}
_DEFAULT_LOCK = threading.Lock()


def _text(value, where: str, limit: int) -> str:
    if not isinstance(value, str):
        raise EngineError(f"{where} must be text.")
    text = unicodedata.normalize("NFC", value).strip()
    if not text:
        raise EngineError(f"{where} is empty.")
    if len(text) > limit:
        raise EngineError(f"{where} is longer than {limit} characters.")
    return text


def _texts(values, what: str, most: int) -> list[str]:
    if not isinstance(values, list) or not values:
        raise EngineError(f"Send a list of {what}.")
    if len(values) > most:
        raise EngineError(f"Send at most {most} {what} per request.")
    return [_text(v, f"{what.capitalize()[:-1]} {i + 1}", LIMITS["text_chars"]) for i, v in enumerate(values)]


def _anchor_sets(value):
    """None for the defaults, or the caller's own sets after checking their shape."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise EngineError("Anchors must be a list of sets, each with five statements.")
    if len(value) > LIMITS["anchor_sets"]:
        raise EngineError(f"Send at most {LIMITS['anchor_sets']} anchor sets.")
    sets = []
    for s, statements in enumerate(value):
        if not isinstance(statements, list) or len(statements) != len(mapping.POINTS):
            raise EngineError(f"Anchor set {s + 1} needs exactly five statements, from 1 (lowest) to 5 (highest).")
        sets.append([_text(t, f"Anchor set {s + 1}, statement {k + 1}", LIMITS["anchor_chars"]) for k, t in enumerate(statements)])
    return sets


def _anchor_vectors(sets, encoder):
    if sets is not None:
        return [encoder.encode(statements) for statements in sets]
    with _DEFAULT_LOCK:
        if encoder.name not in _DEFAULT_VECTORS:
            _DEFAULT_VECTORS[encoder.name] = [encoder.encode(statements) for statements in default_anchors.DEFAULT]
        return _DEFAULT_VECTORS[encoder.name]


def _mappings(value) -> list[str]:
    if value is None:
        return list(mapping.METHODS)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or any(m not in mapping.METHODS for m in value):
        raise EngineError("Mappings must be chosen from: paper, power.")
    return [m for m in mapping.METHODS if m in value]


def _temperatures(value, methods) -> dict:
    value = {} if value is None else value
    if not isinstance(value, dict) or any(k not in mapping.METHODS for k in value):
        raise EngineError("Temperature must be given for paper, power or both.")
    try:
        return {m: mapping.positive(value.get(m, 1.0)) for m in methods}
    except ValueError as exc:
        raise EngineError(str(exc)) from None


def _round(value):
    if isinstance(value, float):
        return round(value, LIMITS["decimals"])
    if isinstance(value, list):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def _about(encoder, custom_anchors: bool) -> dict:
    return {"version": VERSION, "encoder": encoder.name, "dtype": encoder.dtype, "prefix": getattr(encoder, "prefix", ""),
            "meaningful": bool(getattr(encoder, "meaningful", True)), "anchors": "request" if custom_anchors else "default"}


def about(encoder=None) -> dict:
    """What is running. A caller can check this before sending anything."""
    encoder = encoder or encoders.get()
    return {"version": VERSION, "encoder": encoder.name, "dtype": encoder.dtype, "prefix": getattr(encoder, "prefix", ""),
            "device": getattr(encoder, "device", "cpu"),
            "meaningful": bool(getattr(encoder, "meaningful", True)), "mappings": list(mapping.METHODS), "default_anchor_sets": len(default_anchors.DEFAULT),
            "limits": {k: v for k, v in LIMITS.items() if k != "decimals"}}


def _request(request) -> dict:
    if not isinstance(request, dict):
        raise EngineError("The request must be a JSON object.")
    return request


def _similarities(texts, sets, encoder) -> list[list[list[float]]]:
    try:
        anchor_vectors = _anchor_vectors(sets, encoder)
        return [mapping.similarity_sets(vector, anchor_vectors) for vector in encoder.encode(texts)]
    except Exception:
        # A library underneath may put the text it choked on into its own message. That message is
        # dropped here, with its traceback, so it reaches neither the caller nor the Space's logs.
        raise EngineError("The encoder could not score this request.") from None


def score(request, encoder=None) -> dict:
    """{"texts": [...], "anchors": [[5 statements], ...] or absent, "mappings": [...], "temperature": {"paper": 1.0, "power": 1.0}}"""
    request = _request(request)
    unknown = sorted(set(request) - {"texts", "anchors", "mappings", "temperature"})
    if unknown:
        raise EngineError("Unknown field in the request. Allowed: texts, anchors, mappings, temperature.")
    texts = _texts(request.get("texts"), "texts", LIMITS["texts"])
    sets = _anchor_sets(request.get("anchors"))
    methods = _mappings(request.get("mappings"))
    temperatures = _temperatures(request.get("temperature"), methods)
    encoder = encoder or encoders.get()

    results, pmfs = [], {m: [] for m in methods}
    for index, cosines in enumerate(_similarities(texts, sets, encoder)):
        row = {"index": index, "cosines": cosines}
        for method in methods:
            row[method] = mapping.score(cosines, method, temperatures[method])
            pmfs[method].append(row[method]["pmf"])
        results.append(row)
    summary = {m: {"temperature": temperatures[m], "mean_pmf": mapping.mean_pmf(pmfs[m]),
                   "mean_expected_score": mapping.expected_score(mapping.mean_pmf(pmfs[m]))} for m in methods}
    return _round({"engine": _about(encoder, sets is not None), "n": len(texts), "results": results, "summary": summary})


def calibrate(request, encoder=None) -> dict:
    """{"rows": [{"text": ..., "label": 1..5, "split": "calibration" | "holdout" (optional)}, ...], "anchors": ..., "threshold": 3.6 (optional)}"""
    request = _request(request)
    unknown = sorted(set(request) - {"rows", "anchors", "threshold"})
    if unknown:
        raise EngineError("Unknown field in the request. Allowed: rows, anchors, threshold.")
    rows = request.get("rows")
    if not isinstance(rows, list) or len(rows) < 4:
        raise EngineError("Send at least four labelled reactions.")
    if len(rows) > LIMITS["labelled"]:
        raise EngineError(f"Send at most {LIMITS['labelled']} labelled reactions per request.")
    clean = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise EngineError(f"Row {i + 1} must be an object with text and label.")
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in mapping.POINTS:
            raise EngineError(f"Row {i + 1}: label must be a whole number from 1 to 5.")
        split = row.get("split")
        if split not in (None, "calibration", "holdout"):
            raise EngineError(f"Row {i + 1}: split must be calibration or holdout.")
        clean.append({"text": _text(row.get("text"), f"Row {i + 1} text", LIMITS["text_chars"]), "label": label, "split": split})
    threshold = request.get("threshold")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 1 <= threshold <= 5):
        raise EngineError("Threshold must be a number from 1 to 5.")
    sets = _anchor_sets(request.get("anchors"))
    encoder = encoder or encoders.get()

    similarities = _similarities([r["text"] for r in clean], sets, encoder)
    scored = [{"label": r["label"], "split": r["split"], "sets": s} for r, s in zip(clean, similarities, strict=True)]
    try:
        report = calibration.report(scored, threshold)
    except ValueError as exc:
        raise EngineError(str(exc)) from None
    report["rows"] = [{"index": i, "label": r["label"], "split": r["split"], "cosines": r["sets"]} for i, r in enumerate(scored)]
    return _round({"engine": _about(encoder, sets is not None), **report})
