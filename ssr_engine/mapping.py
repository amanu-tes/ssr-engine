"""Semantic similarity rating: a free-text reaction becomes a distribution over a five-point scale.

Two mappings are kept side by side so they can be compared on the same reactions.

paper   Maier et al., "LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of
        Likert Ratings", arXiv:2510.08338, as the authors' reference code computes it
        (github.com/pymc-labs/semantic-similarity-rating, read at commit 86dcd25). For each anchor
        set: g = (1 + cosine) / 2 and p(k) = (g_k - g_min) / sum_j (g_j - g_min). The distributions
        of the sets are then averaged. Temperature, when it is not 1, is p ** (1 / T), renormalised;
        the paper uses T = 1.
power   A simpler variant seen in practice: cosines are averaged per scale point across the anchor
        sets, then p(k) = s_k ** T / sum_j s_j ** T. Nothing is subtracted, so T alone has to pull
        the points apart, and the right T depends heavily on the encoder.

The two differ in three places: the subtraction of the lowest similarity, what is averaged across
anchor sets (distributions or similarities), and the direction of the temperature exponent.
`calibration.py` measures what that does on reactions whose intended point is known.

Pure functions over lists of floats: no model, no third-party import.
"""
from __future__ import annotations

import math

POINTS = (1, 2, 3, 4, 5)
UNIFORM = [1.0 / len(POINTS)] * len(POINTS)
METHODS = ("paper", "power")
T_MIN, T_MAX = 0.001, 10000.0
TEMPERATURE_RULE = "Temperature must be a number from 0.001 to 10000."


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def similarity_sets(reaction, anchor_sets) -> list[list[float]]:
    """Cosine of one reaction vector with every anchor: one list of five per anchor set, in scale order."""
    return checked([[cosine(reaction, anchor) for anchor in anchors] for anchors in anchor_sets])


def checked(sets) -> list[list[float]]:
    """Refuses what would otherwise score as a quiet 3.0: a set that is not five long, or a cosine that is not a number."""
    if not sets or any(len(s) != len(POINTS) for s in sets):
        raise ValueError("Every anchor set needs exactly five statements, one per scale point.")
    if any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) for s in sets for c in s):
        raise ValueError("A similarity is not a finite number; the reaction cannot be scored.")
    return sets


def _power(weights, exponent: float) -> list[float]:
    """Each weight to a power, relative to the largest, so a high exponent cannot round every weight down to zero."""
    top = max(weights)
    if not top > 0:
        return list(weights)
    return [math.exp(exponent * math.log(w / top)) if w > 0 else 0.0 for w in weights]


def _normalise(weights) -> list[float]:
    total = sum(weights)
    if not total > 0 or not math.isfinite(total):
        return list(UNIFORM)       # every point equally similar: the reaction says nothing about the scale
    return [w / total for w in weights]


def positive(temperature) -> float:
    """A temperature this module will use. The range is bounded because outside it the arithmetic stops
    meaning anything: at 1e-310 the exponent is infinite and every reaction would quietly score 3.0."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(TEMPERATURE_RULE)
    value = float(temperature)
    if not math.isfinite(value) or not T_MIN <= value <= T_MAX:
        raise ValueError(TEMPERATURE_RULE)
    return value


def pmf_paper(sets, temperature: float = 1.0, epsilon: float = 0.0) -> list[float]:
    """The paper's mapping. `epsilon` gives the least similar point a little weight (the paper uses 0)."""
    temperature = positive(temperature)
    per_set = []
    for cosines in checked(sets):
        g = [(1.0 + c) / 2.0 for c in cosines]
        low = min(g)
        weights = [x - low for x in g]
        if epsilon > 0:
            weights[g.index(low)] += epsilon
        per_set.append(_normalise(weights))
    mean = [sum(p[k] for p in per_set) / len(per_set) for k in range(len(POINTS))]
    if temperature == 1.0:
        return _normalise(mean)
    return _normalise(_power(mean, 1.0 / temperature))


def pmf_power(sets, temperature: float = 1.0) -> list[float]:
    """The mean-cosine power mapping. A negative mean cosine counts as zero (a power of it is not defined)."""
    temperature = positive(temperature)
    sets = checked(sets)
    mean = [max(0.0, sum(s[k] for s in sets) / len(sets)) for k in range(len(POINTS))]
    return _normalise(_power(mean, temperature))


def expected_score(pmf) -> float:
    return sum(point * p for point, p in zip(POINTS, pmf, strict=True))


def score(sets, method: str, temperature: float = 1.0) -> dict:
    """One reaction under one mapping: its distribution and the mean of it (1.0 to 5.0)."""
    if method == "paper":
        pmf = pmf_paper(sets, temperature)
    elif method == "power":
        pmf = pmf_power(sets, temperature)
    else:
        raise ValueError("Unknown mapping; choose paper or power.")
    return {"pmf": pmf, "expected_score": expected_score(pmf)}


def mean_pmf(pmfs) -> list[float]:
    if not pmfs:
        return list(UNIFORM)
    return _normalise([sum(p[k] for p in pmfs) / len(pmfs) for k in range(len(POINTS))])
