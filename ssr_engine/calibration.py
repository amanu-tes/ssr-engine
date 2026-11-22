"""How well each mapping recovers the scale point a reaction was written for, and the temperature that fits.

A temperature fitted for one encoder is wrong for the next, so it is fitted per run: on the
calibration half of the labelled reactions (lowest mean absolute error against the intended point),
and every figure reported is from the holdout half, which the fit never saw.

Pure functions; rows carry `sets` (the reaction's cosines with each anchor set) and `label` (1 to 5).
"""
from __future__ import annotations

from . import mapping

GRID = {"paper": (0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0),
        "power": (1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)}


def _ranks(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    var = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return cov / var if var else 0.0


def metrics(rows, method: str, temperature: float, threshold: float | None = None) -> dict:
    """One mapping at one temperature on labelled rows. With a threshold: how many reactions written
    for 4 or 5 reach it, and how many written for 1 to 3 reach it when they should not."""
    scored = [(r["label"], mapping.score(r["sets"], method, temperature)) for r in rows]
    expected = [s["expected_score"] for _, s in scored]
    labels = [label for label, _ in scored]
    by_label = {point: [e for label, e in zip(labels, expected, strict=True) if label == point] for point in mapping.POINTS}
    means = {str(point): (sum(v) / len(v) if v else None) for point, v in by_label.items()}
    out = {"mapping": method, "temperature": temperature, "n": len(rows),
           "mae": sum(abs(e - label) for label, e in zip(labels, expected, strict=True)) / len(rows),
           "spearman": spearman(labels, expected),
           "mean_by_label": means,
           "spread": (means["5"] - means["1"]) if means["5"] is not None and means["1"] is not None else None,
           "argmax_accuracy": sum(1 + max(range(5), key=lambda k: s["pmf"][k]) == label for label, s in scored) / len(rows),
           "min": min(expected), "max": max(expected)}
    if threshold is not None:
        high = [e for label, e in zip(labels, expected, strict=True) if label >= 4]
        low = [e for label, e in zip(labels, expected, strict=True) if label <= 3]
        out.update(high_reaching=sum(e >= threshold for e in high), high_total=len(high),
                   low_reaching=sum(e >= threshold for e in low), low_total=len(low))
    return out


def fit(rows, method: str) -> float:
    """The grid temperature with the lowest mean absolute error; ties go to the lower temperature."""
    return float(min(GRID[method], key=lambda t: (metrics(rows, method, t)["mae"], t)))


def assign_splits(rows) -> None:
    """Rows without a split alternate within each label, first to calibration, so both halves see every point."""
    seen: dict[int, int] = {}
    for row in rows:
        if row.get("split") not in ("calibration", "holdout"):
            count = seen.get(row["label"], 0)
            row["split"] = "calibration" if count % 2 == 0 else "holdout"
            seen[row["label"]] = count + 1


def report(rows, threshold: float | None = None) -> dict:
    assign_splits(rows)
    calibration = [r for r in rows if r["split"] == "calibration"]
    holdout = [r for r in rows if r["split"] == "holdout"]
    if not calibration or not holdout:
        raise ValueError("Send at least two reactions for a label, so the fitting half and the holdout half both have some.")
    fitted = {method: fit(calibration, method) for method in mapping.METHODS}
    table = []
    for method in mapping.METHODS:
        table.append({"setting": f"{method}, T = 1", **metrics(holdout, method, 1.0, threshold)})
        table.append({"setting": f"{method}, fitted T", **metrics(holdout, method, fitted[method], threshold)})
    return {"fitted_temperature": fitted, "calibration_n": len(calibration), "holdout_n": len(holdout), "holdout": table}
