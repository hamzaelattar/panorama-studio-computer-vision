"""Mise en correspondance KNN et test du ratio de Lowe."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np


@dataclass
class MatchResult:
    knn_matches: list
    good_matches: list
    query_count: int
    pairs_with_two_neighbors: int
    elapsed_ms: float
    error: str | None = None


def match_descriptors(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
    norm_type: int,
    ratio: float = 0.75,
) -> MatchResult:
    """Associe deux ensembles de descripteurs sans planter sur les petits jeux."""

    start = perf_counter()
    if descriptors_a is None or descriptors_b is None:
        return MatchResult([], [], 0, 0, 0.0, "Descripteurs absents dans au moins une image.")
    if len(descriptors_a) == 0 or len(descriptors_b) < 2:
        return MatchResult(
            [], [], int(len(descriptors_a)), 0, (perf_counter() - start) * 1000.0,
            "Il faut au moins un descripteur source et deux descripteurs cibles.",
        )
    if not 0.0 < ratio < 1.0:
        raise ValueError("Le ratio de Lowe doit être strictement compris entre 0 et 1.")

    # SIFT attend float32 ; les descripteurs binaires restent uint8.
    if norm_type == cv2.NORM_L2:
        descriptors_a = np.asarray(descriptors_a, dtype=np.float32)
        descriptors_b = np.asarray(descriptors_b, dtype=np.float32)
    else:
        descriptors_a = np.asarray(descriptors_a, dtype=np.uint8)
        descriptors_b = np.asarray(descriptors_b, dtype=np.uint8)

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    valid_pairs = [neighbors for neighbors in knn if len(neighbors) >= 2]
    good = [m for m, n in valid_pairs if m.distance < ratio * n.distance]
    return MatchResult(
        knn_matches=knn,
        good_matches=good,
        query_count=len(knn),
        pairs_with_two_neighbors=len(valid_pairs),
        elapsed_ms=(perf_counter() - start) * 1000.0,
    )
