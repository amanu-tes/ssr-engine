"""Default anchor statements: three ways of wording the same five-point scale of intent to act.

Each set runs from 1 (will not) to 5 (will). Averaging over several wordings keeps one phrasing's
quirks from deciding the score. They are deliberately generic; send your own with a request when
your question is not "would this person go ahead?".
"""
from __future__ import annotations

DEFAULT = [
    [
        "I would definitely not go ahead with this.",
        "I probably would not go ahead with this.",
        "I am not sure whether I would go ahead with this.",
        "I would probably go ahead with this.",
        "I would definitely go ahead with this.",
    ],
    [
        "This is not for me at all and I have no interest in it.",
        "I have little interest in this.",
        "I am neutral about this; it could go either way.",
        "This interests me and I am inclined to say yes.",
        "This is exactly what I want and I will say yes.",
    ],
    [
        "There is no chance I act on this.",
        "It is unlikely that I act on this.",
        "I might or might not act on this.",
        "It is likely that I act on this.",
        "I am certain to act on this.",
    ],
]
