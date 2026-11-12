"""Frozen synthetic corpus shared by the Gaussian and Ising toy pilots.

The family labels belong to the generator and evaluation manifest only.  Model
inputs contain token IDs, never these labels or hand-assigned coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import numpy as np


POOLS = {
    "F": ("money", "loan", "credit", "deposit", "finance", "lender", "payment", "cash"),
    "R": ("river", "stream", "shore", "flood", "water", "creek", "sediment", "current"),
    "M": ("melody", "rhythm", "singer", "guitar", "piano", "chorus", "concert", "lyric"),
}
CENTERS = {"F": "bank", "R": "bank", "M": "song"}


@dataclass(frozen=True)
class Window:
    family: str
    tokens: tuple[str, str, str]


def vocabulary() -> tuple[str, ...]:
    return ("bank", "song", *(token for pool in POOLS.values() for token in pool))


def _split_pair_indices() -> tuple[list[int], list[int], list[int]]:
    pairs = list(combinations(range(8), 2))
    rng = np.random.default_rng(1730)
    while True:
        permutation = rng.permutation(len(pairs)).tolist()
        train, validation, test = permutation[:16], permutation[16:22], permutation[22:]
        coverage = {anchor: 0 for anchor in range(8)}
        for index in train:
            for anchor in pairs[index]:
                coverage[anchor] += 1
        if min(coverage.values()) >= 2:
            return train, validation, test


def split_windows() -> dict[str, tuple[Window, ...]]:
    pair_indices = _split_pair_indices()
    partitions = {"train": [], "validation": [], "test": []}
    for family, anchors in POOLS.items():
        pairs = list(combinations(anchors, 2))
        for name, indices in zip(partitions, pair_indices):
            partitions[name].extend(Window(family, (CENTERS[family], *pairs[index])) for index in indices)
    return {name: tuple(rows) for name, rows in partitions.items()}


def manifest() -> dict[str, object]:
    splits = split_windows()
    content = {name: [{"family": row.family, "tokens": row.tokens} for row in rows] for name, rows in splits.items()}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"vocabulary": vocabulary(), "splits": content, "sha256": hashlib.sha256(encoded).hexdigest()}
